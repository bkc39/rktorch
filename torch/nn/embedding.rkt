#lang racket/base

;; nn.Embedding with standard-normal weight init, matching
;; nn.Embedding.reset_parameters (init.normal_) so a shared seed yields
;; parameters bit-comparable to PyTorch. #:padding-idx is deliberately
;; unsupported (a documented gap vs. the PyTorch reference).

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
