#lang racket/base

(require (only-in racket/generic define/generic)
         (only-in racket/match match-define)
         (only-in "../../nn.rkt" define-layer gen:layer layer-copy layer-rebuild))

(provide copied-field
         (struct-out deleg)
         holder
         holder?
         keep
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

(define (keep _name c)
  c)

(struct deleg ([inner #:mutable] via)
  #:methods gen:layer
  [(define/generic raw-rebuild layer-rebuild-method)
   (define (layer-forward self . _inputs) (deleg-inner self))
   (define (layer-rebuild-method self child tensor)
     (match-define (deleg inner via) self)
     (deleg (case via
              [(plain) (layer-rebuild inner child tensor)]
              [(generic) (raw-rebuild inner child tensor)]
              [(fixed) (layer-rebuild inner keep values)])
            via))])
