#lang racket/base

;; Runner + tests for the literate example ../../examples/racket/02-matmul.rkt.

(require torchrkt
         "../racket/02-matmul.rkt")

(module+ main
  (define t (run-example))
  (printf "shape: ~a\n" (tensor-shape t))
  (display (tensor->string t))
  (newline))

(module+ test
  (require rackunit)
  (define t (run-example))
  (check-equal? (tensor-shape t) '(2 2))
  ;; a = [[0 1 2] [3 4 5]]; a . a^T = [[5 14] [14 50]].
  (check-equal? (tensor->list t) '(5.0 14.0 14.0 50.0)))
