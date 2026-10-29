#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign/device-type.rkt" device)
           (only-in "../foreign/raw/accounts.rkt"
                    account-of note-accounted! note-unaccounted! since-sample-bytes)
           (only-in (submod "../foreign/raw/accounts.rkt" private) accounts)
           (only-in "../foreign/raw/cas.rkt" cas-add! cas-update!)
           (only-in "../foreign/raw/collector.rkt" finalizer-runs)
           (only-in "../foreign/raw/device-queries.rkt" allocator-reading)
           (only-in "../foreign/raw/memory.rkt"
                    finalizer-diagnostics finalizer-failures record-allocation!
                    swallow-and-count-failure unaccount!)
           (only-in "../foreign/raw/pressure.rkt" collect-under-pressure! native-memory-limit)
           (only-in "../foreign/raw/regions.rkt" with-uninterruptible)
           (only-in "private/race-harness.rkt" join settle! spawn with-race-hook)
           (only-in "private/within.rkt" within))

  (define mib (* 1024 1024))

  (define (forget-account! dev)
    (cas-update! accounts (lambda (all) (hash-remove all dev))))

  (define (wait-until done? seconds)
    (define until (+ (current-inexact-milliseconds) (* 1000 seconds)))
    (let poll ()
      (unless (or (done?) (> (current-inexact-milliseconds) until))
        (sleep 0.001)
        (poll))))

  (define (spin-until done? seconds)
    (define until (+ (current-inexact-milliseconds) (* 1000 seconds)))
    (let spin ()
      (unless (or (done?) (> (current-inexact-milliseconds) until))
        (spin))))

  (define (two-samples-at-once dev)
    (define a (account-of dev))
    (define mark (* 64 mib))
    (define accounted mib)
    (define arrived (box 0))
    (define reading-calls (box 0))
    (define (both-at-gate name subject)
      (when (and (eq? name 'gate-read) (eq? subject dev))
        (cas-add! arrived 1)
        (wait-until (lambda () (>= (unbox arrived) 2)) 10)))
    (define (held-reading d)
      (cond
        [(eq? d dev)
         (define n (cas-add! reading-calls 1))
         (if (= n 2)
             (note-accounted! a accounted)
             (wait-until (lambda () (>= (unbox reading-calls) 2)) 10))
         (* 16 mib)]
        [else #f]))
    (note-accounted! a (* 8 mib))
    (with-race-hook both-at-gate
      (parameterize ([native-memory-limit mark]
                     [allocator-reading held-reading])
        (define samplers
          (for/list ([_ (in-range 2)])
            (spawn (lambda () (collect-under-pressure! dev)) #:pool 'own)))
        (for-each join samplers)))
    (begin0
      (list (unbox reading-calls) (since-sample-bytes a))
      (note-unaccounted! a (* 9 mib))
      (forget-account! dev)))

  (test-case "two samples at once take each byte once"
    (within 60
            (lambda ()
              (define-values (calls since-sample)
                (apply values (two-samples-at-once (device 'cuda 18))))
              (check-equal? calls 2 "the two threads did not both sample")
              (check-true (>= since-sample mib)
                          (format "~a bytes left toward the next sample, of ~a accounted after both took"
                                  since-sample mib)))))

  (test-case "a drain polling from a parallel thread never counts a run before its release"
    (within 60
            (lambda ()
              (define holding (box #f))
              (define polled (box #f))
              (define poller
                (spawn (lambda ()
                         (wait-until (lambda () (unbox holding)) 10)
                         (set-box! polled (finalizer-runs)))
                       #:pool 'own))
              (define release-held
                (swallow-and-count-failure
                 (lambda (_h)
                   (set-box! holding #t)
                   (spin-until (lambda () (unbox polled)) 10))))
              (define before
                (with-uninterruptible
                  (begin0 (finalizer-runs) (release-held 'handle))))
              (join poller)
              (check-eqv? (unbox polled) before "the run counted before its release finished")
              (check-eqv? (finalizer-runs) (add1 before) "the finished run went uncounted"))))

  (test-case "a release that raises still counts its run and its failure"
    (define raising (swallow-and-count-failure (lambda (_h) (raise 'released-badly))))
    (define-values (runs failures)
      (with-uninterruptible
        (define runs (finalizer-runs))
        (define failures (finalizer-failures))
        (raising 'handle)
        (values (- (finalizer-runs) runs) (- (finalizer-failures) failures))))
    (check-equal? (list runs failures) '(1 1)))

  (define (ledger-entries)
    (cdr (assq 'ledger-entries (finalizer-diagnostics))))

  (test-case "the ledger's entry count is kept, not walked"
    (within 120
            (lambda ()
              (settle!)
              (define dev (device 'cuda 19))
              (define n 200000)
              (define keys (for/list ([i (in-range n)]) (box i)))
              (define start (ledger-entries))
              (for ([k (in-list keys)]) (record-allocation! k 64 dev))
              (define held (ledger-entries))
              (define-values (_results _cpu real _gc)
                (time-apply (lambda () (for ([_ (in-range 100)]) (finalizer-diagnostics))) '()))
              (for ([k (in-list keys)]) (unaccount! k))
              (define after (ledger-entries))
              (forget-account! dev)
              (check-equal? (- held start) n)
              (check-equal? after start)
              (check-true (< real 100)
                          (format "100 diagnostics over ~a entries took ~a ms" n real))))))
