#lang racket/base

;; Loss functions, pure Racket over the op tranche.

(require (only-in "../foreign.rkt" mean mul sub to-dtype)
         ;; generated but uncontracted; this layer supplies the typed surface
         (only-in "../generated.rkt" [cross-entropy-loss g:cross-entropy-loss]))

(provide mse-loss
         cross-entropy)

;; torch.nn.functional.mse_loss with the default mean reduction.
(define (mse-loss prediction target)
  (define d (sub prediction target))
  (mean (mul d d)))

;; F.cross_entropy defaults: mean reduction, no class weighting,
;; ignore_index -100, label_smoothing 0; targets coerced to int64.
(define (cross-entropy logits targets)
  (g:cross-entropy-loss logits (to-dtype targets 'int64) #f 1 -100 0.0))
