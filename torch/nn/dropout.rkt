#lang racket/base

;; nn.Dropout (inverted dropout; identity in eval mode). Hand-written
;; gen:module rather than define-module because it carries mutable mode
;; state, flipped by the train!/eval! protocol.

(require (only-in "../generated.rkt" dropout)
         (only-in "module.rkt"
                  gen:module
                  module-forward
                  module-parameters
                  module-named-parameters
                  module-buffers
                  module-set-training!
                  module-training?))

(provide Dropout
         dropout?)

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

(define (Dropout #:p [p 0.5])
  (Dropout% p #t))

(define dropout? Dropout%?)
