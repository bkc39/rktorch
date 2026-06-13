#lang racket/base

;; Runner + tests for the literate example ../../examples/racket/04-mlp.rkt.

(require (except-in racket/list argmax flatten)
         torch
         ;; conv2d/max-pool2d/flatten name both the functional torch ops and
         ;; the nn layers; this MLP uses neither, so drop the nn layer names
         ;; to avoid the require collision.
         (except-in torch/nn conv2d max-pool2d flatten)
         "../racket/04-mlp.rkt")

(module+ main
  (define-values (losses net) (run-example))
  (printf "losses: ~a\n" losses)
  (for ([nm+p (in-list (named-parameters net))])
    (printf "~a: ~a\n" (car nm+p) (tensor-shape (cdr nm+p)))))

(module+ test
  (require rackunit)
  (define-values (losses net) (run-example))
  (check-equal? (length losses) 5)
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (check-equal? (map car (named-parameters net))
                '("fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias"))
  (check-equal? (map tensor-shape (parameters net))
                '((8 4) (8) (2 8) (2))))
