#lang racket/base

(require (only-in racket/contract/base
                  ->* ->i </c >=/c and/c contract flat-contract-predicate or/c)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * / < current-generator device dtype generator? rand-like tensor?
                  to-dtype)
         (only-in "../foreign/generator.rkt" generator-on?)
         (only-in "../foreign/ops.rkt" any-float-dtype/c)
         (prefix-in g: (only-in "../generated.rkt" dropout))
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "layer.rkt" define-layer training? with-mode))

(define float-dtype? (flat-contract-predicate any-float-dtype/c))

(define (drawing p training? generator)
  (and (supplied-or training? #t)
       (positive? (supplied-or p 0.5))
       (or (supplied-or generator) (current-generator))))

(define dropout/c
  (->i ([x tensor?])
       (#:p [p (and/c real? (>=/c 0) (</c 1))]
        #:training? [training? boolean?]
        #:generator [generator (or/c generator? #f)])
       #:pre/name (x p training? generator)
       "the generator lives on the tensor's device"
       (let ([g (drawing p training? generator)])
         (or (not g) (generator-on? g (device x))))
       #:pre/name (x p training? generator)
       "a floating-point tensor when drawing from a generator"
       (or (not (drawing p training? generator)) (float-dtype? (dtype x)))
       [result tensor?]))

(define/contract-out (dropout x #:p [p 0.5] #:training? [training? #t] ;; noqa
                              #:generator [generator #f])
  dropout/c
  (define drawn-from (drawing p training? generator))
  (cond
    [drawn-from
     (define keep (- 1.0 p))
     (* x (~> (rand-like x #:generator drawn-from) (< keep) (to-dtype (dtype x)) (/ keep)))]
    [else (g:dropout x (exact->inexact p) training?)]))

(define layer-dropout (contract dropout/c dropout 'Dropout 'caller 'Dropout #f))

(define-layer Dropout (p) ;; noqa
  #:contract (->* [] [#:p (and/c (>=/c 0) (</c 1))] dropout?)
  #:init (#:p [p 0.5])
  #:forward (x #:generator [generator #f])
  (with-mode
    (layer-dropout x #:p p #:training? (training? mode) #:generator generator)))
