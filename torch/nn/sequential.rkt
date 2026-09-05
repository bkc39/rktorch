#lang racket/base

(require (only-in racket/contract/base ->* list/c listof or/c)
         (only-in "module.rkt" define-layer in-layers layer? LayerList))

(define step/c (or/c layer? procedure?))

(define-layer Sequential (layers) ;; noqa
  #:contract (->* [] #:rest (or/c (list/c (listof step/c)) (listof step/c))
                  sequential?)
  #:init (#:rest steps)
  (set! layers (LayerList (if (and (pair? steps) (list? (car steps)))
                              (car steps)
                              steps)
                          #:prefix ""))
  #:forward (x)
  (for/fold ([acc x])
            ([m (in-layers layers)])
    (m acc)))
