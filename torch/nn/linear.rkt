#lang racket/base

(require (only-in "../foreign.rkt" add matmul transpose)
         (only-in "init.rkt" kaiming-uniform uniform-init)
         (only-in "module.rkt" define-module))

(provide Linear
         (rename-out [Linear? linear?]) ;; noqa
         )

(define-module Linear (in-features out-features)
  ;; weight before bias: nn.Linear.reset_parameters' RNG draw order
  #:params ([weight (kaiming-uniform (list out-features in-features))]
            [bias (let ([bound (/ 1.0 (sqrt in-features))])
                    (uniform-init (list out-features) (- bound) bound))])
  #:forward (x)
  (add (matmul x (transpose weight 0 1)) bias))
