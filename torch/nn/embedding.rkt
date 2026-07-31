#lang racket/base

;; nn.Embedding: a learned lookup table. The weight is [num-embeddings,
;; embedding-dim] with standard-normal init, matching
;; nn.Embedding.reset_parameters (init.normal_) so a shared seed yields
;; parameters bit-comparable to PyTorch. Forward defers to the functional
;; `embedding` on the facade (F.embedding arg order: indices first).
;; #:padding-idx is deferred — nanoGPT-style capstones don't need it, and
;; nn.Embedding's zero-the-row init interplay deserves its own treatment.

(require (only-in "../foreign.rkt" embedding)
         (only-in "init.rkt" normal-init)
         (only-in "module.rkt" define-module))

;; PascalCase constructor, lowercase predicate (the conv.rkt convention);
;; embedding? aliases the struct predicate Embedding%? (noqa: raco review
;; can't see the alias).
(provide Embedding
         embedding? ;; noqa
         )

(define-module Embedding% (num-embeddings embedding-dim)
  #:params ([weight (normal-init (list num-embeddings embedding-dim))])
  #:reflection-name 'Embedding
  #:forward (indices)
  (embedding indices weight))

(define (Embedding num-embeddings embedding-dim)
  (Embedding% num-embeddings embedding-dim))

(define embedding? Embedding%?)
