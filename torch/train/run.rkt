#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre id))
         (only-in racket/contract/base
                  -> ->* <=/c >/c and/c any any/c contract-out hash/c listof
                  not/c or/c real-in)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt" device?)
         (only-in "../nn/optim.rkt" optimizer?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/types.rkt"
                  define-named-contract not-implemented scheduler-config?
                  trainer?))

(provide (contract-out [run? (-> any/c boolean?)]))

(struct run ())

(define-syntax-parse-rule (define-run-contracts name:id ...)
  (begin (define-named-contract name run?) ...))

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

(define-run-contracts
  fitting-run/c batch-start-run/c step-pending-run/c logging-run/c
  manual-run/c)

(define/contract-out (run-trainer r)
  (-> run? trainer?)
  (not-implemented 'run-trainer 'T1))

(define/contract-out (run-stage r)
  (-> run? (or/c #f run-stage/c))
  (not-implemented 'run-stage 'T1))

(define/contract-out (run-epoch r)
  (-> run? exact-nonnegative-integer?)
  (not-implemented 'run-epoch 'T1))

(define/contract-out (run-global-step r)
  (-> run? exact-nonnegative-integer?)
  (not-implemented 'run-global-step 'T1))

(define/contract-out (run-batch-index r)
  (-> run? (or/c #f exact-nonnegative-integer?))
  (not-implemented 'run-batch-index 'T1))

(define/contract-out (run-optimizers r)
  (-> run? (listof optimizer?))
  (not-implemented 'run-optimizers 'T1))

(define/contract-out (run-device r)
  (-> run? device?)
  (not-implemented 'run-device 'T1))

(define/contract-out (run-stepped? r)
  (-> run? boolean?)
  (not-implemented 'run-stepped? 'T1))

(define/contract-out (run-metric r key #:default [default #f])
  (->* (run? symbol?) (#:default any/c) any)
  (not-implemented 'run-metric 'T1))

(define/contract-out (run-metrics r)
  (-> run? (hash/c symbol? real? #:immutable #t))
  (not-implemented 'run-metrics 'T1))

(define/contract-out (request-stop! r)
  (-> fitting-run/c void?)
  (not-implemented 'request-stop! 'T1))

(define/contract-out (skip-step! r)
  (-> step-pending-run/c void?)
  (not-implemented 'skip-step! 'T1))

(define/contract-out (skip-epoch! r)
  (-> batch-start-run/c void?)
  (not-implemented 'skip-epoch! 'T1))

(define/contract-out (run-schedulers r)
  (-> run? (listof scheduler-config?))
  (not-implemented 'run-schedulers 'T2))

(define/contract-out (run-estimated-stepping-batches r)
  (-> run? exact-nonnegative-integer?)
  (not-implemented 'run-estimated-stepping-batches 'T2))
