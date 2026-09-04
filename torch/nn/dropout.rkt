#lang racket/base

(require (only-in racket/contract/base -> ->* </c >=/c and/c any/c)
         (only-in "../generated.rkt" dropout)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "module.rkt"
                  gen:layer
                  layer-buffers
                  layer-forward
                  layer-named-children
                  layer-named-parameters
                  layer-parameters
                  layer-set-training!
                  layer-training?))

(struct Dropout% (p [training? #:mutable])
  #:reflection-name 'Dropout
  #:property prop:procedure
  (lambda (self . inputs) (apply layer-forward self inputs))
  #:methods gen:layer
  [(define (layer-forward self . inputs)
     (apply (lambda (x)
              (dropout x (Dropout%-p self) (Dropout%-training? self)))
            inputs))
   (define (layer-parameters self) '())
   (define (layer-named-parameters self prefix) '())
   (define (layer-buffers self) '())
   (define (layer-named-children self) '())
   (define (layer-set-training! self training?)
     (set-Dropout%-training?! self training?))
   (define (layer-training? self)
     (Dropout%-training? self))])

(define/contract-out (Dropout #:p [p 0.5]) ;; noqa
  (->* [] [#:p (and/c (>=/c 0) (</c 1))] dropout?)
  (Dropout% p #t))

(define/contract-out dropout? (-> any/c boolean?) Dropout%?)
