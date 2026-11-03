#lang racket/base

(require (only-in "../../foreign.rkt" tensor?)
         (only-in "layer-containers.rkt"
                  Children% Children%-alist Children%? Parameters% Parameters%-alist
                  Parameters%?)
         (only-in "layer-core.rkt" layer?))

(provide rebuilt-value)

(define (rebuilt-value name v child tensor) ;; noqa
  (let walk ([v v])
    (cond
      [(layer? v) (child name v)]
      [(tensor? v) (tensor v)]
      [(Parameters%? v)
       (Parameters% (for/list ([e (in-list (Parameters%-alist v))])
                      (cons (car e) (tensor (cdr e)))))]
      [(Children%? v)
       (Children% (for/list ([e (in-list (Children%-alist v))])
                    (cons (car e) (child (car e) (cdr e)))))]
      [(pair? v) (cons (walk (car v)) (walk (cdr v)))]
      [(vector? v)
       (define copy (for/vector #:length (vector-length v)
                                ([x (in-vector v)])
                      (walk x)))
       (if (immutable? v) (vector->immutable-vector copy) copy)]
      [(box? v) (box (walk (unbox v)))]
      [else v])))
