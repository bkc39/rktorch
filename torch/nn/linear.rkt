#lang racket/base

;; nn.Linear: y = x W^T + b. Init mirrors nn.Linear.reset_parameters —
;; kaiming uniform a=sqrt5 for the weight, then bias uniform on
;; +/-1/sqrt(fan_in), in that order — so a shared seed matches PyTorch.

(require (only-in "../foreign.rkt" add matmul transpose)
         (only-in "init.rkt" kaiming-uniform uniform-init)
         (only-in "module.rkt" define-module))

(provide Linear
         (rename-out [Linear? linear?]) ;; noqa
         )

(define-module Linear (in-features out-features)
  #:params ([weight (kaiming-uniform (list out-features in-features))]
            [bias (let ([bound (/ 1.0 (sqrt in-features))])
                    (uniform-init (list out-features) (- bound) bound))])
  #:forward (x)
  (add (matmul x (transpose weight 0 1)) bias))
