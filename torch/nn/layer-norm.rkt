#lang racket/base

;; nn.LayerNorm: normalization over the trailing normalized-shape dims with
;; learned affine parameters — weight starts at ones, bias at zeros
;; (nn.LayerNorm.reset_parameters; deterministic, no RNG draw). Forward
;; defers to the functional `layer-norm` on the facade.

(require (only-in "../foreign.rkt" layer-norm ones zeros)
         (only-in "module.rkt" define-module))

;; PascalCase constructor, lowercase predicate (the linear.rkt convention);
;; layer-norm? doesn't collide with the functional layer-norm — the
;; differing `?` keeps `(require torch torch/nn)` clean. LayerNorm/
;; LayerNorm? are define-module expansions, invisible to raco review.
(provide LayerNorm
         (rename-out [LayerNorm? layer-norm?]) ;; noqa
         )

;; An int names the trailing dim, a list the trailing dims — nn.LayerNorm.
(define-module LayerNorm (normalized-shape #:eps [eps 1e-5])
  #:coerce ([normalized-shape (if (list? normalized-shape)
                                  normalized-shape
                                  (list normalized-shape))])
  #:params ([weight (apply ones normalized-shape)]
            [bias (apply zeros normalized-shape)])
  #:forward (x)
  (layer-norm x normalized-shape #:weight weight #:bias bias #:eps eps))
