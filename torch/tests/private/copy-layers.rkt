#lang racket/base

(require (only-in "../../nn.rkt" define-layer layer-copy))

(provide copied-field
         holder
         holder?
         twin
         twin?)

(define-layer holder (slot)
  #:init (v)
  (set! slot v)
  #:forward ()
  slot)

(define-layer twin (a b)
  #:init (x y)
  (set! a x)
  (set! b y)
  #:forward ()
  (list a b))

(define (copied-field m)
  ((layer-copy m)))
