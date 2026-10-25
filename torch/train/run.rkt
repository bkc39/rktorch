#lang racket/base

(require (only-in racket/contract/base
                  -> ->* <=/c >/c and/c any any/c contract? contract-out
                  flat-named-contract hash/c listof not/c or/c real-in)
         (only-in "../foreign.rkt" device?)
         (only-in "../nn/optim.rkt" optimizer?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/types.rkt"
                  not-implemented scheduler-config? trainer?))

(provide (contract-out [run? (-> any/c boolean?)]))

(struct run () #:name run-struct #:constructor-name new-run)

(define/contract-out stage/c contract?
  (flat-named-contract 'stage/c (or/c 'fit 'test 'predict)))

(define/contract-out run-stage/c contract?
  (flat-named-contract 'run-stage/c
                       (or/c 'train 'sanity-check 'validate 'test 'predict)))

(define/contract-out limit/c contract?
  (flat-named-contract
   'limit/c
   (or/c exact-nonnegative-integer? (and/c inexact-real? (real-in 0.0 1.0)))))

(define/contract-out val-check-interval/c contract?
  (flat-named-contract
   'val-check-interval/c
   (or/c #f exact-positive-integer? (and/c inexact-real? (>/c 0.0) (<=/c 1.0)))))

(define/contract-out precision/c contract?
  (flat-named-contract 'precision/c (or/c '32-true '16-mixed 'bf16-mixed)))

(define/contract-out clip-algorithm/c contract?
  (flat-named-contract 'clip-algorithm/c (or/c 'norm 'value)))

(define/contract-out fast-dev-run/c contract?
  (flat-named-contract 'fast-dev-run/c (or/c boolean? exact-positive-integer?)))

(define/contract-out interval/c contract?
  (flat-named-contract 'interval/c (or/c 'step 'epoch)))

(define/contract-out state-key/c contract?
  (flat-named-contract 'state-key/c (and/c string? (not/c ""))))

(define/contract-out fitting-run/c contract?
  (flat-named-contract 'fitting-run/c run?))

(define/contract-out batch-start-run/c contract?
  (flat-named-contract 'batch-start-run/c run?))

(define/contract-out step-pending-run/c contract?
  (flat-named-contract 'step-pending-run/c run?))

(define/contract-out logging-run/c contract?
  (flat-named-contract 'logging-run/c run?))

(define/contract-out manual-run/c contract?
  (flat-named-contract 'manual-run/c run?))

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
