#lang racket/base

(require (only-in racket/contract/base -> cons/c listof)
         (only-in "layer.rkt"
                  child-name/c children-by-key define-layer step/c))

(define-layer LayerHash (items) ;; noqa
  #:contract (-> (listof (cons/c child-name/c step/c)) layer-hash?)
  #:init (entries)
  (set! items (children-by-key entries))
  #:forward (_x)
  (raise-arguments-error 'LayerHash
                         "not applicable; look up a child with child-ref"))
