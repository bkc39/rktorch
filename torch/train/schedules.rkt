#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i >=/c non-empty-listof or/c)
         (only-in "private/types.rkt" callback? define-named-contract)
         (only-in "run.rkt" interval/c)
         (only-in "trainer.rkt" define-callback define-init))

(define-named-contract schedulable-key/c
  (or/c 'lr 'weight-decay 'momentum 'beta1 'beta2 'eps 'alpha))

(define real-curve/c (-> exact-nonnegative-integer? real?))

(define nonnegative-curve/c (-> exact-nonnegative-integer? (>=/c 0)))

(define (curve/c key)
  (if (memq key '(beta1 beta2 eps)) real-curve/c nonnegative-curve/c))

(define-callback hparam-scheduler (name curve interval)
  #:contract (->* (symbol? (-> exact-nonnegative-integer? real?)) (#:interval interval/c)
                  callback?)
  #:kind plain
  (define-init (n c #:interval [i 'step])
    (error 'unimplemented)))

(define-callback param-group-scheduler (key curve optimizer groups interval)
  #:contract (->i ([key schedulable-key/c] [curve (key) (curve/c key)])
                  (#:optimizer [optimizer exact-nonnegative-integer?]
                   #:groups [groups (or/c #f (non-empty-listof exact-nonnegative-integer?))]
                   #:interval [interval interval/c])
                  [_ callback?])
  #:kind plain
  (define-init (k c #:optimizer [o 0] #:groups [gs #f] #:interval [i 'step])
    (error 'unimplemented)))
