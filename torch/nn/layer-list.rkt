#lang racket/base

(require (only-in racket/contract/base -> listof)
         (only-in "module.rkt" children-by-index define-layer step/c))

(define-layer LayerList (items) ;; noqa
  #:contract (-> (listof step/c) layer-list?)
  #:init (layers)
  (set! items (children-by-index layers))
  #:forward (_x)
  (raise-arguments-error 'LayerList "not applicable; iterate with in-layers"))
