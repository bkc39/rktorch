#lang racket/base

(require (only-in racket/contract/base -> ->* </c >=/c and/c any/c)
         (only-in "../generated.rkt" dropout)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "module.rkt"
                  gen:module
                  module-forward
                  module-parameters
                  module-named-parameters
                  module-buffers
                  module-set-training!
                  module-training?))

(struct Dropout% (p [training? #:mutable])
  #:reflection-name 'Dropout
  #:property prop:procedure
  (lambda (self . inputs) (apply module-forward self inputs))
  #:methods gen:module
  [(define (module-forward self . inputs)
     (apply (lambda (x)
              (dropout x (Dropout%-p self) (Dropout%-training? self)))
            inputs))
   (define (module-parameters self) '())
   (define (module-named-parameters self prefix) '())
   (define (module-buffers self) '())
   (define (module-set-training! self training?)
     (set-Dropout%-training?! self training?))
   (define (module-training? self)
     (Dropout%-training? self))])

(define/contract-out (Dropout #:p [p 0.5]) ;; noqa
  (->* [] [#:p (and/c (>=/c 0) (</c 1))] dropout?)
  (Dropout% p #t))

(define/contract-out dropout? (-> any/c boolean?) Dropout%?)
