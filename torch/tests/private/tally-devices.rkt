#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../foreign/device-type.rkt" device)
         (only-in "../../foreign/raw/accounts.rkt" account-of note-mark!)
         (only-in (submod "../../foreign/raw/accounts.rkt" private) accounts raw-live)
         (only-in "../../foreign/raw/cas.rkt" cas-update!)
         (only-in "../../foreign/raw/tallies.rkt" flush-tallies!)
         (only-in (submod "../../foreign/raw/tallies.rkt" private)
                  registered-tallies tally-owner))

(provide kib
         mib
         mark
         shared-live
         tally-of-thread
         with-batched-device
         with-unmarked-device)

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
