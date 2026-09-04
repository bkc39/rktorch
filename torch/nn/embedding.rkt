#lang racket/base

(require (only-in racket/contract/base ->)
         (only-in "../foreign.rkt" embedding)
         (only-in "init.rkt" normal-init)
         (only-in "module.rkt" define-module))

(define-module Embedding (num-embeddings embedding-dim) ;; noqa
  #:contract (-> exact-positive-integer? exact-positive-integer? embedding?)
  #:params ([weight (normal-init (list num-embeddings embedding-dim))])
  #:forward (indices)
  (embedding indices weight))
