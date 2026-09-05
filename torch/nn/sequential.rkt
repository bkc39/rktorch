#lang racket/base

(require (only-in racket/contract/base -> or/c)
         (only-in "module.rkt" define-layer in-layers layer? LayerList))

(define-layer Sequential (layers) ;; noqa
  #:contract (-> (or/c layer? procedure?) ... sequential?)
  #:init (#:rest ms)
  (set! layers (LayerList ms #:prefix ""))
  #:forward (x)
  (for/fold ([acc x])
            ([m (in-layers layers)])
    (m acc)))
