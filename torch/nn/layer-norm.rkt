#lang racket/base

(require (only-in racket/contract/base ->* non-empty-listof or/c)
         (only-in "../foreign.rkt" layer-norm ones zeros)
         (only-in "module.rkt" define-module))

(define-module LayerNorm (normalized-shape #:eps [eps 1e-5]) ;; noqa
  #:contract (->* [(or/c exact-positive-integer?
                         (non-empty-listof exact-positive-integer?))]
                  [#:eps real?]
                  layer-norm?)
  #:coerce ([normalized-shape (if (list? normalized-shape)
                                  normalized-shape
                                  (list normalized-shape))])
  #:params ([weight (apply ones normalized-shape)]
            [bias (apply zeros normalized-shape)])
  #:forward (x)
  (layer-norm x normalized-shape #:weight weight #:bias bias #:eps eps))
