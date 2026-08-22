#lang racket/base

(require (only-in "../foreign.rkt" embedding)
         (only-in "init.rkt" normal-init)
         (only-in "module.rkt" define-module))

(provide Embedding
         (rename-out [Embedding? embedding?]) ;; noqa
         )

(define-module Embedding (num-embeddings embedding-dim)
  #:params ([weight (normal-init (list num-embeddings embedding-dim))])
  #:forward (indices)
  (embedding indices weight))
