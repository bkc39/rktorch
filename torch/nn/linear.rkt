#lang racket/base

(require (only-in racket/contract/base ->*)
         (only-in "../foreign.rkt" linear)
         (only-in "init.rkt" kaiming-uniform uniform-bias)
         (only-in "layer.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer Linear (weight bias) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:bias? boolean?]
                  linear?)
  #:init (in-features out-features #:bias? [bias? #t])
  ;; weight before bias: nn.Linear.reset_parameters' RNG draw order
  (set! weight (Parameter (kaiming-uniform (list out-features in-features))))
  (set! bias (and bias? (Parameter (uniform-bias out-features in-features))))
  #:forward (x)
  (linear x weight #:bias bias))
