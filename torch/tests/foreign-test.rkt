#lang racket/base

(module+ test
  (require rackunit
           ffi/vector
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/unsafe/ops unsafe-struct*-ref)
           (only-in "../foreign/structs.rkt"
                    tensor-impl-shape tensor-state-index)
           (submod "../foreign.rkt" unsafe)
           "../main.rkt")

  (define-runtime-path structs-module "../foreign/structs.rkt")

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
    (check-exn exn:fail:contract? (lambda () (tensor-free! t))))

  (test-case "a fresh tensor's state field, found by its derived index, is live"
    (check-eq? (unsafe-struct*-ref (zeros 2 2) tensor-state-index) 'live))

  (test-case "a tensor cannot be chaperoned"
    (check-exn #rx"authentic"
               (lambda ()
                 (chaperone-struct (zeros 2 2) tensor-impl-shape
                                   (lambda (_ v) v)))))

  (test-case "the wrapper's state has no exported setter"
    (define-values (variables _syntax) (module->exports structs-module))
    (define names (for*/list ([phase+names (in-list variables)]
                              [entry (in-list (cdr phase+names))])
                    (car entry)))
    (check-not-false (memq 'tensor-impl-state names))
    (check-false (memq 'set-tensor-impl-state! names)))

  (test-case "a view still reads its values after its base is freed"
    (define base (tensor '((1.0 2.0 3.0) (4.0 5.0 6.0))))
    (define row (narrow base 0 1 1))
    (tensor-free! base)
    (check-equal? (tensor->list row) '(4.0 5.0 6.0))))
