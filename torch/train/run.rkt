#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre expr id))
         (only-in racket/contract/base
                  -> ->* <=/c >/c and/c any any/c contract-out hash/c listof
                  not/c or/c real-in)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt" device?)
         (only-in "../nn/optim.rkt" optimizer?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/book.rkt" book-metric book-metrics)
         (only-in "private/hooks.rkt" hook-log hook-named)
         (only-in "private/state.rkt"
                  live? run-batch-index run-book run-current-stage run-device run-entry
                  run-epoch run-global-step run-hook run-loop run-optimizers run-step-pending?
                  run-stepped? run-trainer run? set-run-skip-epoch?! set-run-skip-window?!
                  set-run-stop?!)
         (only-in "private/types.rkt"
                  define-named-contract not-implemented scheduler-config?
                  trainer-automatic? trainer?))

(provide (contract-out
          [run? (-> any/c boolean?)]
          [run-trainer (-> run? trainer?)]
          [run-epoch (-> run? exact-nonnegative-integer?)]
          [run-global-step (-> run? exact-nonnegative-integer?)]
          [run-batch-index (-> run? (or/c #f exact-nonnegative-integer?))]
          [run-optimizers (-> run? (listof optimizer?))]
          [run-device (-> run? device?)]
          [run-stepped? (-> run? boolean?)]))

(module+ private
  (provide hook-log-rule))

(define (hook-log-rule r)
  (define name (and (live? r) (run-hook r)))
  (and name (hook-log (hook-named name))))

(define-syntax-parse-rule (define-live-run-contract name:id (r:id) test:expr)
  (define-named-contract name (lambda (r) (and (live? r) test))))

(define-named-contract stage/c (or/c 'fit 'test 'predict))

(define-named-contract run-stage/c
  (or/c 'train 'sanity-check 'validate 'test 'predict))

(define-named-contract limit/c
  (or/c exact-nonnegative-integer? (and/c inexact-real? (real-in 0.0 1.0))))

(define-named-contract val-check-interval/c
  (or/c #f exact-positive-integer? (and/c inexact-real? (>/c 0.0) (<=/c 1.0))))

(define-named-contract precision/c (or/c '32-true '16-mixed 'bf16-mixed))

(define-named-contract clip-algorithm/c (or/c 'norm 'value))

(define-named-contract fast-dev-run/c (or/c boolean? exact-positive-integer?))

(define-named-contract interval/c (or/c 'step 'epoch))

(define-named-contract state-key/c (and/c string? (not/c "")))

(define-live-run-contract fitting-run/c (r)
  (eq? (run-entry r) 'fit))

(define-live-run-contract batch-start-run/c (r)
  (eq? (run-hook r) 'on-train-batch-start))

(define-live-run-contract step-pending-run/c (r)
  (run-step-pending? r))

(define-named-contract logging-run/c
  (lambda (r) (and (hook-log-rule r) #t)))

(define-live-run-contract manual-run/c (r)
  (and (eq? (run-hook r) 'training-step)
       (not (trainer-automatic? (run-trainer r)))))

(define/contract-out (run-stage r)
  (-> run? (or/c #f run-stage/c))
  (and (live? r) (run-current-stage r)))

(define/contract-out (run-metric r key #:default [default #f])
  (->* (run? symbol?) (#:default any/c) any)
  (book-metric (run-book r) (run-loop r) key default))

(define/contract-out (run-metrics r)
  (-> run? (hash/c symbol? real? #:immutable #t))
  (book-metrics (run-book r)))

(define/contract-out (request-stop! r)
  (-> fitting-run/c void?)
  (set-run-stop?! r #t))

(define/contract-out (skip-step! r)
  (-> step-pending-run/c void?)
  (set-run-skip-window?! r #t))

(define/contract-out (skip-epoch! r)
  (-> batch-start-run/c void?)
  (set-run-skip-epoch?! r #t))

(define/contract-out (run-schedulers r)
  (-> run? (listof scheduler-config?))
  (not-implemented 'run-schedulers 'T2))

(define/contract-out (run-estimated-stepping-batches r)
  (-> run? exact-nonnegative-integer?)
  (not-implemented 'run-estimated-stepping-batches 'T2))
