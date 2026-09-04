#lang racket/base

(require (only-in racket/contract/base ->)
         (only-in "../foreign.rkt" embedding)
         (only-in "init.rkt" normal-init)
         (only-in "module.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer Embedding (weight) ;; noqa
  #:contract (-> exact-positive-integer? exact-positive-integer? embedding?)
  #:init (num-embeddings embedding-dim)
  (set! weight (Parameter (normal-init (list num-embeddings embedding-dim))))
  #:forward (indices)
  (embedding indices weight))
