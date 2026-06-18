#lang racket/base

;; The first layer: y = x W^T + b, defined through define-module exactly as a
;; user would write it. Init mirrors nn.Linear.reset_parameters — kaiming
;; uniform with a = sqrt 5 for the weight, then uniform on +/- 1/sqrt(fan_in)
;; for the bias — in that order, so a shared seed yields identical
;; parameters to PyTorch.

(require (only-in "../foreign.rkt" add matmul transpose)
         (only-in "init.rkt" kaiming-uniform uniform-init)
         (only-in "module.rkt" define-module))

;; nn layer constructors are PascalCase, mirroring the torch.nn.* class names
;; (and keeping them distinct from the lowercase functional ops on `torch`).
;; Linear? is defined by the define-module expansion, invisible to raco review.
(provide Linear
         Linear? ;; noqa
         )

(define-module Linear (in-features out-features)
  #:params ([weight (kaiming-uniform (list out-features in-features))]
            [bias (let ([bound (/ 1.0 (sqrt in-features))])
                    (uniform-init (list out-features) (- bound) bound))])
  #:forward (x)
  (add (matmul x (transpose weight 0 1)) bias))
