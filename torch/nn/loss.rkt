#lang racket/base

;; Loss functions, pure Racket over the op tranche.

(require (only-in "../foreign.rkt" mean mul sub))

(provide mse-loss)

;; torch.nn.functional.mse_loss with the default mean reduction.
(define (mse-loss prediction target)
  (define d (sub prediction target))
  (mean (mul d d)))
