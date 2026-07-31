#lang racket/base

;; nn.LayerNorm: normalization over the trailing normalized-shape dims with
;; learned affine parameters — weight starts at ones, bias at zeros
;; (nn.LayerNorm.reset_parameters; deterministic, no RNG draw). Forward
;; defers to the functional `layer-norm` on the facade.

(require (only-in "../foreign.rkt" layer-norm ones zeros)
         (only-in "module.rkt" define-module))

;; PascalCase constructor, lowercase predicate (the conv.rkt convention).
;; layer-norm? aliases the struct predicate LayerNorm%? and doesn't collide
;; with the functional layer-norm — the differing `?` keeps `(require torch
;; torch/nn)` clean (noqa: raco review can't see the alias).
(provide LayerNorm
         layer-norm? ;; noqa
         )

;; normalized-shape arrives already normalized to a list from the smart
;; constructor, so the parameter shapes are straightforward.
(define-module LayerNorm% (normalized-shape eps)
  #:params ([weight (apply ones normalized-shape)]
            [bias (apply zeros normalized-shape)])
  #:reflection-name 'LayerNorm
  #:forward (x)
  (layer-norm x normalized-shape #:weight weight #:bias bias #:eps eps))

;; An int names the trailing dim, a list the trailing dims — nn.LayerNorm.
(define (LayerNorm normalized-shape #:eps [eps 1e-5])
  (LayerNorm% (if (list? normalized-shape)
                  normalized-shape
                  (list normalized-shape))
              eps))

(define layer-norm? LayerNorm%?)
