#lang racket/base

(require (only-in "../foreign.rkt" mean mul sub to-dtype)
         (only-in "../generated.rkt" [cross-entropy-loss g:cross-entropy-loss]))

(provide mse-loss
         cross-entropy)

(define (mse-loss prediction target)
  (define d (sub prediction target))
  (mean (mul d d)))

(define (cross-entropy logits targets)
  (g:cross-entropy-loss logits (to-dtype targets 'int64) #f 1 -100 0.0))
