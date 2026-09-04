#lang racket/base

(require (only-in racket/contract/base ->)
         (only-in "module.rkt"
                  LayerList define-layer layer-list->list layer?))

;; #:prefix "" registers the list under no name of its own, so the children
;; are "0.weight", "1.bias", ... as in nn.Sequential
(define-layer Sequential (layers) ;; noqa
  #:contract (-> layer? ... sequential?)
  #:init (#:rest ms)
  (set! layers (LayerList ms #:prefix ""))
  #:forward (x)
  (for/fold ([acc x]) ([m (in-list (layer-list->list layers))])
    (m acc)))
