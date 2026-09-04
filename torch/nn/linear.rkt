#lang racket/base

(require (only-in racket/contract/base ->)
         (only-in "../foreign.rkt" add matmul transpose)
         (only-in "init.rkt" kaiming-uniform uniform-init)
         (only-in "module.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer Linear (weight bias) ;; noqa
  #:contract (-> exact-positive-integer? exact-positive-integer? linear?)
  #:init (in-features out-features)
  ;; weight before bias: nn.Linear.reset_parameters' RNG draw order
  (set! weight (Parameter (kaiming-uniform (list out-features in-features))))
  (set! bias
        (let ([bound (/ 1.0 (sqrt in-features))])
          (Parameter (uniform-init (list out-features) (- bound) bound))))
  #:forward (x)
  (add (matmul x (transpose weight 0 1)) bias))
