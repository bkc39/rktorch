#lang racket/base

;; Runner + tests for the literate example ../../examples/racket/03-autograd.rkt.

(require torchrkt
         "../racket/03-autograd.rkt")

(module+ main
  (define g (run-example))
  (printf "grad: ~a\n" (tensor->list g)))

(module+ test
  (require rackunit)
  (define g (run-example))
  (check-equal? (tensor-shape g) '(3))
  ;; d(sum(x^2))/dx = 2x at x = [1 2 3]
  (check-equal? (tensor->list g) '(2.0 4.0 6.0)))
