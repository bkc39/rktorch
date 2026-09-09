#lang racket/base

(require (only-in racket/contract/base ->* list/c listof or/c)
         (only-in "layer.rkt"
                  children-by-index define-layer in-layers layer-forward
                  step/c))

(define-layer Sequential (steps) ;; noqa
  #:contract (->* [] #:rest (or/c (list/c (listof step/c)) (listof step/c))
                  sequential?)
  #:init (#:rest ms)
  (set! steps (children-by-index (if (and (pair? ms) (list? (car ms)))
                                     (car ms)
                                     ms)))
  #:forward (x)
  (for/fold ([acc x])
            ([m (in-layers steps)])
    (layer-forward m acc)))
