#lang racket/base

(require (only-in racket/contract/base ->)
         (only-in "module.rkt" define-layer in-layers layer? LayerList))

(define-layer Sequential (layers) ;; noqa
  #:contract (-> layer? ... sequential?)
  #:init (#:rest ms)
  (set! layers (LayerList ms #:prefix ""))
  #:forward (x)
  (for/fold ([acc x])
            ([m (in-layers layers)])
    (m acc)))
