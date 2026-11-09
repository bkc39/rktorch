#lang racket/base

(require (only-in "../../nn.rkt" define-layer layer-copy layer-rebuild))

(provide copied-field
         holder
         holder?
         rebuild-directly
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

(define rebuild-directly layer-rebuild)
