#lang racket/base

;; Runner + tests for the literate example ../../examples/racket/01-arith.rkt.

(require torch
         "../racket/01-arith.rkt")

(module+ main
  (define t (run-example))
  (printf "shape: ~a\n" (tensor-shape t))
  (display (tensor->string t))
  (newline))

(module+ test
  (require rackunit)
  (define t (run-example))
  (check-equal? (tensor-shape t) '(2 2))
  ;; (x + 1) * relu(x) over [[1 -2] [3 -4]]: relu zeroes the negatives, and
  ;; the negative (x + 1) factor makes those zeros IEEE -0.0.
  (check-equal? (tensor->list t) '(2.0 -0.0 12.0 -0.0)))
