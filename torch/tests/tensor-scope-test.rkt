#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt" + item ones sum tensor?)
           (only-in "../foreign/raw/scope.rkt" scope-diagnostics)
           (only-in "../foreign/scope.rkt" tensor-keep with-tensor-scope))

  (define (exit-frees) (hash-ref (scope-diagnostics) 'exit-frees))

  (test-case "a temporary the scope made is freed at its exit; the result escapes"
    (define leaked #f)
    (define before (exit-frees))
    (define kept
      (with-tensor-scope
        (define t (ones 2 2))
        (set! leaked (+ t 1))
        (+ leaked 1)))
    (check-true (tensor? kept))
    (check-equal? (item (sum kept)) 12.0)
    (check-false (tensor? leaked))
    (check-exn exn:fail:contract? (lambda () (sum leaked)))
    (check-equal? (- (exit-frees) before) 2))

  (test-case "a nested scope's result is owned by its parent"
    (define inner #f)
    (with-tensor-scope
      (set! inner (with-tensor-scope (ones 3)))
      (check-true (tensor? inner))
      (void))
    (check-false (tensor? inner)))

  (test-case "a non-local exit frees everything the scope owns"
    (define made #f)
    (check-exn #rx"boom"
               (lambda ()
                 (with-tensor-scope
                   (set! made (ones 2))
                   (error "boom"))))
    (check-false (tensor? made)))

  (test-case "tensor-keep moves a temporary out of the scope"
    (define kept #f)
    (with-tensor-scope
      (set! kept (tensor-keep (ones 4)))
      (void))
    (check-true (tensor? kept))
    (check-equal? (item (sum kept)) 4.0)))
