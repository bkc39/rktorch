#lang racket/base

(require (only-in "../foreign.rkt" layer-norm ones zeros)
         (only-in "module.rkt" define-module))

(provide LayerNorm
         (rename-out [LayerNorm? layer-norm?]) ;; noqa
         )

(define-module LayerNorm (normalized-shape #:eps [eps 1e-5])
  #:coerce ([normalized-shape (if (list? normalized-shape)
                                  normalized-shape
                                  (list normalized-shape))])
  #:params ([weight (apply ones normalized-shape)]
            [bias (apply zeros normalized-shape)])
  #:forward (x)
  (layer-norm x normalized-shape #:weight weight #:bias bias #:eps eps))
