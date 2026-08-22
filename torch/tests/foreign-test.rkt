#lang racket/base

(module+ test
  (require rackunit
           ffi/vector
           (submod "../foreign.rkt" unsafe)
           "../main.rkt")

  (test-case "version looks like a torch version"
    (define v (torch-version))
    (check-true (string? v))
    (check-true (regexp-match? #rx"^[0-9]+[.][0-9]+" v) v))

  (test-case "randn has the requested shape and element count"
    (manual-seed! 0)
    (define t (randn 2 2))
    (check-true (tensor? t))
    (check-equal? (tensor-shape t) '(2 2))
    (check-equal? (tensor-numel t) 4)
    (check-equal? (length (tensor->list t)) 4)
    (check-equal? (f32vector-length (tensor->vector t)) 4))

  (test-case "seeding is deterministic"
    (manual-seed! 0)
    (define a (tensor->list (randn 2 2)))
    (manual-seed! 0)
    (define b (tensor->list (randn 2 2)))
    (check-equal? a b))

  (test-case "different seeds give different draws"
    (manual-seed! 0)
    (define a (tensor->list (randn 2 2)))
    (manual-seed! 1)
    (define b (tensor->list (randn 2 2)))
    (check-not-equal? a b))

  (test-case "higher-rank shapes"
    (manual-seed! 7)
    (define t (randn 3 4))
    (check-equal? (tensor-shape t) '(3 4))
    (check-equal? (tensor-numel t) 12))

  (test-case "tensor->string renders something"
    (manual-seed! 0)
    (check-true (> (string-length (tensor->string (randn 2 2))) 0)))

  (test-case "tensor->repr matches PyTorch's REPL form"
    ;; seed 0: the canonical PyTorch randn(2,2)
    (manual-seed! 0)
    (check-equal? (tensor->repr (randn 2 2))
                  "tensor([[ 1.5410, -0.2934],\n        [-2.1788,  0.5684]])"))

  (test-case "explicit free is idempotent at the contract boundary"
    (manual-seed! 0)
    (define t (randn 2 2))
    (tensor-free! t)
    (check-exn exn:fail:contract? (lambda () (tensor-free! t)))))
