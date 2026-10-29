#lang racket/base

(module+ test
  (require (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../foreign.rkt" tensor? zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/accelerator.rkt" make-mps-probe mps-usable?)
           (only-in "../foreign/device-type.rkt" device)
           (only-in "../foreign/raw/cas.rkt" cas-claim! cas-update!)
           (only-in "../foreign/raw/collector.rkt" call-as-the-collector)
           (only-in "../foreign/raw/memory.rkt" finalizer-diagnostics)
           (only-in "../foreign/raw/pressure.rkt" account-of)
           (only-in (submod "../foreign/raw/accounts.rkt" private) accounts)
           (only-in "../private/lock.rkt" make-lock with-lock)
           (only-in "private/race-harness.rkt"
                    check-ledger gate-hook ledger-snapshot make-gate
                    release-gate! settle! wait-for-arrival)
           (only-in "private/spurious-cas.rkt" call-with-spurious-cas-failures))

  (define (runs) (cdr (assq 'runs (finalizer-diagnostics))))

  (test-case "the CAS loops retry a spurious failure"
    (define b (box 0))
    (check-equal? (call-with-spurious-cas-failures
                   '(update) (lambda () (cas-update! b add1)))
                  1)
    (check-equal? (unbox b) 1)
    (define c (box 'live))
    (check-true (call-with-spurious-cas-failures
                 '(claim) (lambda () (cas-claim! c 'live 'mine))))
    (check-eq? (unbox c) 'mine))

  (test-case "an explicit free retries its state, lifetime and charge claims"
    (settle!)
    (define base (ledger-snapshot))
    (define t (zeros 64 64))
    (call-with-spurious-cas-failures
     '(tensor-state lifetime claim update) (lambda () (tensor-free! t)))
    (check-false (tensor? t) "the free did not release the handle")
    (check-ledger base #f))

  (test-case "the finalizer retries its lifetime and charge claims"
    (settle!)
    (define base (ledger-snapshot))
    (define before (runs))
    (void (zeros 64 64))
    (call-with-spurious-cas-failures '(lifetime claim update) settle!)
    (check-true (> (runs) before) "the finalizer released nothing")
    (check-ledger base #f))

  (test-case "the collector's claim retries"
    (check-eq? (call-with-spurious-cas-failures
                '(collector)
                (lambda () (call-as-the-collector (lambda () 'collected))))
               'collected))

  (test-case "a device's first account retries its insert"
    (define dev (device 'cuda 13))
    (define a (call-with-spurious-cas-failures '(update) (lambda () (account-of dev))))
    (check-eq? (account-of dev) a)
    (void (cas-update! accounts (lambda (all) (hash-remove all dev)))))

  (test-case "the MPS probe's claim retries"
    (check-equal? (call-with-spurious-cas-failures
                   '(probe) (lambda () (mps-usable? (make-mps-probe void))))
                  (mps-usable? (make-mps-probe void))))

  (test-case "the library lock's acquire and release retry"
    (define lock (make-lock))
    (check-eq? (call-with-spurious-cas-failures
                '(lock) (lambda () (with-lock lock 'held)))
               'held)
    (check-eq? (with-lock lock 'again) 'again "a release left the lock held"))

  (test-case "the race harness's gate retries its take"
    (define g (make-gate 'spurious-point))
    (define parker
      (thread
       (lambda ()
         (call-with-spurious-cas-failures
          '(gate) (lambda () ((gate-hook g) 'spurious-point 'subject))))))
    (check-eq? (wait-for-arrival g #:seconds 10) 'parked)
    (release-gate! g)
    (check-not-false (sync/timeout 10 parker) "the parked thread never left the gate"))

  (define (expands? form)
    (parameterize ([current-namespace (make-base-namespace)])
      (with-handlers ([exn:fail:syntax? (lambda (_) #f)])
        (expand form)
        #t)))

  (define-runtime-path cas-module "../foreign/raw/cas.rkt")

  (define (site-program site)
    `(module m racket/base
       (require (file ,(path->string cas-module)))
       (cas-update! (box 0) add1 #:site ',site)))

  (test-case "a CAS site outside the enumeration is a syntax error"
    (check-true (expands? (site-program 'update)))
    (check-false (expands? (site-program 'updat)) "a misspelt site expanded")))
