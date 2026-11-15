#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../foreign/device-type.rkt" device)
         (only-in "../../foreign/raw/accounts.rkt" account-of note-mark!)
         (only-in (submod "../../foreign/raw/accounts.rkt" private) accounts raw-live)
         (only-in "../../foreign/raw/cas.rkt" cas-update!)
         (only-in "../../foreign/raw/memory.rkt" record-allocation! unaccount!)
         (only-in "../../foreign/raw/tallies.rkt" flush-tallies! flush-tally!)
         (only-in (submod "../../foreign/raw/tallies.rkt" private)
                  registered-tallies tally-owner)
         (only-in "race-harness.rkt" join spawn with-race-hook worker-thread))

(provide kib
         mib
         mark
         shared-live
         tally-of-thread
         with-batched-device
         with-unmarked-device
         unaccount-all!
         charging
         with-parked
         call-with-held-charge
         collections-of
         spin-until)

(define kib 1024)
(define mib (* 1024 kib))
(define mark (* 512 mib))

(define (shared-live dev) (raw-live (account-of dev)))

(define (tally-of-thread th)
  (memf (lambda (t) (eq? (weak-box-value (tally-owner t)) th)) (registered-tallies)))

(define (forget-account! dev)
  (flush-tallies!)
  (void (cas-update! accounts (lambda (all) (hash-remove all dev)))))

(define (call-with-fake-device index proc #:mark [device-mark mark])
  (define dev (device 'cuda index))
  (dynamic-wind
   (lambda ()
     (when device-mark
       (note-mark! (account-of dev) device-mark))
     (flush-tallies!))
   (lambda () (proc dev))
   (lambda () (forget-account! dev))))

(define-syntax-parse-rule (with-batched-device (dev:id index:expr) body:expr ...+)
  (call-with-fake-device index (lambda (dev) body ...)))

(define-syntax-parse-rule (with-unmarked-device (dev:id index:expr) body:expr ...+)
  (call-with-fake-device index (lambda (dev) body ...) #:mark #f))

(define (unaccount-all! keys)
  (for-each unaccount! keys)
  (flush-tallies!))

(define ((charging dev keys nbytes))
  (for ([k (in-list keys)])
    (record-allocation! k nbytes dev)))

(define (call-with-parked work proc #:pool [pool 'own])
  (define done (make-semaphore 0))
  (define hold (make-semaphore 0))
  (define w
    (spawn (lambda ()
             (work)
             (semaphore-post done)
             (semaphore-wait hold))
           #:pool pool))
  (semaphore-wait done)
  (dynamic-wind
   void
   (lambda () (proc w))
   (lambda ()
     (semaphore-post hold)
     (unless (thread-dead? (worker-thread w))
       (join w)))))

(define-syntax-parse-rule (with-parked (w:id work:expr
                                             (~optional (~seq #:pool pool:expr)
                                                        #:defaults ([pool #''own])))
                            body:expr ...+)
  (call-with-parked work (lambda (w) body ...) #:pool pool))

(define (call-with-held-charge dev nbytes thunk)
  (flush-tally!)
  (define key (box 'held))
  (begin0
    (with-parked (_w (charging dev (list key) nbytes))
      (thunk))
    (unaccount-all! (list key))))

(define (collections-of dev thunk)
  (define resets (box 0))
  (define (count-resets name subject)
    (when (and (eq? name 'checks-resetting) (equal? subject dev))
      (set-box! resets (add1 (unbox resets)))))
  (with-race-hook count-resets
    (thunk))
  (unbox resets))

(define (spin-until ready? #:seconds [seconds 20])
  (define deadline (+ (current-milliseconds) (* 1000 seconds)))
  (let spin ()
    (cond
      [(ready?) #t]
      [(> (current-milliseconds) deadline) #f]
      [else (spin)])))
