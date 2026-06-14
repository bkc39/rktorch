#lang racket/base

;; Loss functions, pure Racket over the op tranche.

(require (only-in "../foreign.rkt" mean mul sub to-dtype)
         ;; cross_entropy_loss is generated but uncontracted; the nn layer is
         ;; where it gets a typed, defaulted surface.
         (only-in "../generated.rkt" [cross-entropy-loss g:cross-entropy-loss]))

(provide mse-loss
         cross-entropy)

;; torch.nn.functional.mse_loss with the default mean reduction.
(define (mse-loss prediction target)
  (define d (sub prediction target))
  (mean (mul d d)))

;; torch.nn.functional.cross_entropy with the default mean reduction: raw
;; logits in, integer class-index targets (coerced to int64 via the to-dtype
;; bridge), no class weighting, ignore_index -100, label_smoothing 0.
(define (cross-entropy logits targets)
  (g:cross-entropy-loss logits (to-dtype targets 'int64) #f 1 -100 0.0))
