#lang racket/base

;; nn.Dropout: during training, zero each activation independently with
;; probability p and scale the survivors by 1/(1-p) (inverted dropout); in
;; eval it's the identity. The generated functional op already honors its
;; train flag, so forward just threads the layer's mode through. dropout is a
;; hand-written gen:module (not define-module) because it carries mutable
;; mode state, flipped by the train!/eval! protocol.

(require (only-in "../generated.rkt" [dropout f:dropout])
         (only-in "module.rkt"
                  gen:module
                  module-forward
                  module-parameters
                  module-named-parameters
                  module-buffers
                  module-set-training!
                  module-training?))

;; dropout?/the constructor are the public names; the struct is dropout-impl.
(provide dropout
         dropout?)

;; Modules default to training mode, like torch.nn (call eval! to switch).
(struct dropout-impl (p [training? #:mutable])
  #:reflection-name 'dropout
  #:property prop:procedure
  (lambda (self . inputs) (apply module-forward self inputs))
  #:methods gen:module
  [(define (module-forward self . inputs)
     (apply (lambda (x)
              (f:dropout x (dropout-impl-p self) (dropout-impl-training? self)))
            inputs))
   ;; dropout has no learnable params or buffers.
   (define (module-parameters self) '())
   (define (module-named-parameters self prefix) '())
   (define (module-buffers self) '())
   (define (module-set-training! self training?)
     (set-dropout-impl-training?! self training?))
   (define (module-training? self)
     (dropout-impl-training? self))])

(define (dropout #:p [p 0.5])
  (dropout-impl p #t))

(define dropout? dropout-impl?)
