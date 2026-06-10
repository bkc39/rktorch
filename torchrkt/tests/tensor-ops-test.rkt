#lang racket/base

;; Unit tests for the v1 op tranche (creation, shape, elementwise, reductions,
;; linalg, marshalling). Value parity with PyTorch lives in python-cross-test;
;; these pin the Racket-facing behavior.

(module+ test
  (require rackunit
           "../main.rkt")

  (test-case "creation goldens"
    (check-equal? (tensor->list (zeros 2 2)) '(0.0 0.0 0.0 0.0))
    (check-equal? (tensor->list (ones 3)) '(1.0 1.0 1.0))
    (check-equal? (tensor->list (full 7.5 2)) '(7.5 7.5))
    (check-equal? (tensor->list (arange 3)) '(0.0 1.0 2.0))
    (check-equal? (tensor->list (arange 1 2.5 0.5)) '(1.0 1.5 2.0))
    (check-equal? (tensor->list (eye 2)) '(1.0 0.0 0.0 1.0))
    (check-equal? (tensor-shape (eye 2 3)) '(2 3)))

  (test-case "tensor from nested lists infers the shape"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor-shape t) '(2 3))
    (check-equal? (tensor->list t) '(1.0 2.0 3.0 4.0 5.0 6.0))
    (check-equal? (tensor-shape (tensor 5)) '())
    (check-equal? (tensor-shape (tensor '(1 2))) '(2))
    (check-exn exn:fail? (lambda () (tensor '((1 2) (3))))))

  (test-case "shape ops"
    (define t (tensor '((1 2 3) (4 5 6))))
    (check-equal? (tensor-shape (reshape t 3 2)) '(3 2))
    (check-equal? (tensor-shape (reshape t -1)) '(6))
    (check-equal? (tensor-shape (view t 6)) '(6))
    (check-equal? (tensor->list (transpose t 0 1)) '(1.0 4.0 2.0 5.0 3.0 6.0))
    (check-equal? (tensor-shape (permute t 1 0)) '(3 2))
    (check-equal? (tensor-shape (unsqueeze t 0)) '(1 2 3))
    (check-equal? (tensor-shape (squeeze (unsqueeze t 0))) '(2 3))
    (check-equal? (tensor-shape (squeeze (unsqueeze t 0) 0)) '(2 3))
    (check-equal? (tensor-shape (cat (list t t))) '(4 3))
    (check-equal? (tensor-shape (cat (list t t) 1)) '(2 6))
    (check-equal? (tensor-shape (stack (list t t))) '(2 2 3)))

  (test-case "elementwise with scalar dispatch on either side"
    (define x (tensor '(1 -2 3)))
    (check-equal? (tensor->list (add x x)) '(2.0 -4.0 6.0))
    (check-equal? (tensor->list (add x 1)) '(2.0 -1.0 4.0))
    (check-equal? (tensor->list (add 1 x)) '(2.0 -1.0 4.0))
    (check-equal? (tensor->list (sub x 1)) '(0.0 -3.0 2.0))
    (check-equal? (tensor->list (sub 1 x)) '(0.0 3.0 -2.0))
    (check-equal? (tensor->list (mul x 2)) '(2.0 -4.0 6.0))
    (check-equal? (tensor->list (div x 2)) '(0.5 -1.0 1.5))
    (check-equal? (tensor->list (div 6 (tensor '(1 2 3)))) '(6.0 3.0 2.0))
    (check-equal? (tensor->list (neg x)) '(-1.0 2.0 -3.0))
    (check-equal? (tensor->list (relu x)) '(1.0 0.0 3.0))
    (check-equal? (tensor->list (pow x 2)) '(1.0 4.0 9.0)))

  (test-case "exp/log/sqrt/max/min fall back to racket/base on numbers"
    (check-equal? (exp 0) 1)
    (check-equal? (log 1) 0)
    (check-equal? (log 8 2) 3.0)
    (check-equal? (sqrt 4) 2)
    (check-equal? (max 1 2 3) 3)
    (check-equal? (min 1 2 3) 1)
    (define x (tensor '(1 4 9)))
    (check-equal? (tensor->list (sqrt x)) '(1.0 2.0 3.0))
    (check-= (item (exp (tensor 0))) 1.0 1e-6)
    (check-= (item (log (tensor 1))) 0.0 1e-6)
    (check-equal? (item (max x)) 9.0)
    (check-equal? (item (min x)) 1.0))

  (test-case "reductions"
    (define t (tensor '((1 2) (3 4))))
    (check-equal? (item (sum t)) 10.0)
    (check-equal? (item (mean t)) 2.5)
    (check-equal? (item (argmax t)) 3.0)
    (check-equal? (tensor->list (argmax t 1)) '(1.0 1.0))
    (check-equal? (tensor-shape (argmax t 1 #:keepdim #t)) '(2 1))
    (define p (softmax t 1))
    (define vals (tensor->list p))
    (check-= (+ (car vals) (cadr vals)) 1.0 1e-6)
    (check-= (item (sum (exp (log-softmax t 1)))) 2.0 1e-5))

  (test-case "linalg"
    (define a (tensor '((1 2) (3 4))))
    (define v (tensor '(1 1)))
    (check-equal? (tensor->list (matmul a a)) '(7.0 10.0 15.0 22.0))
    (check-equal? (tensor->list (mm a a)) '(7.0 10.0 15.0 22.0))
    (check-equal? (tensor->list (mv a v)) '(3.0 7.0))
    (check-equal? (item (dot v v)) 2.0)
    ;; shape mismatch surfaces as a Racket error carrying the C++ message
    (check-exn exn:fail? (lambda () (mv a (tensor '(1 1 1))))))

  (test-case "item and to-dtype"
    (check-equal? (item (tensor 42)) 42.0)
    (check-exn exn:fail? (lambda () (item (tensor '(1 2)))))
    (define i (to-dtype (tensor '(1.5 2.5)) 'int64))
    (check-equal? (tensor->list i) '(1.0 2.0)))

  (test-case "rand and uniform! stay in range"
    (manual-seed! 0)
    (define u (rand 64))
    (for ([x (in-list (tensor->list u))])
      (check-true (and (>= x 0.0) (< x 1.0))))
    (define w (zeros 64))
    (uniform! w -2.0 -1.0)
    (for ([x (in-list (tensor->list w))])
      (check-true (and (>= x -2.0) (< x -1.0))))))
