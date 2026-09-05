#lang racket/base

(require (only-in racket/contract/base ->* non-empty-listof or/c)
         (only-in "../foreign.rkt" layer-norm ones zeros)
         (only-in "module.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer LayerNorm (normalized-shape eps weight bias) ;; noqa
  #:contract (->* [(or/c exact-positive-integer?
                         (non-empty-listof exact-positive-integer?))]
                  [#:eps real?]
                  layer-norm?)
  #:init (normalized-shape #:eps [eps 1e-5])
  (set! normalized-shape
        (if (list? normalized-shape) normalized-shape (list normalized-shape)))
  (set! weight (Parameter (apply ones normalized-shape)))
  (set! bias (Parameter (apply zeros normalized-shape)))
  #:forward (x)
  (layer-norm x normalized-shape #:weight weight #:bias bias #:eps eps))
