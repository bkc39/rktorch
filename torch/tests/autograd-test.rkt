#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt")

  (test-case "grad of sum(x*x) is 2x"
    (define x (requires-grad! (tensor '(1.0 2.0 3.0))))
    (check-true (requires-grad? x))
    (backward! (sum (mul x x)))
    (check-equal? (tensor->list (grad x)) '(2.0 4.0 6.0)))

  (test-case "grad before backward errors"
    (define x (requires-grad! (tensor '(1.0 2.0))))
    (check-exn exn:fail? (lambda () (grad x))))

  (test-case "with-no-grad suspends recording and restores the mode"
    (check-true (grad-enabled?))
    (define x (requires-grad! (tensor '(1.0 2.0))))
    (define y (with-no-grad (mul x x)))
    (check-true (grad-enabled?))
    (check-exn exn:fail? (lambda () (backward! (sum y)))))

  (test-case "with-no-grad restores the mode on escape"
    (check-exn exn:fail?
               (lambda () (with-no-grad (error 'boom "escape"))))
    (check-true (grad-enabled?)))

  (test-case "detach leaves the graph"
    (define x (requires-grad! (tensor '(1.0 2.0))))
    (define d (detach (mul x x)))
    (check-false (requires-grad? d))
    (check-equal? (tensor->list d) '(1.0 4.0)))

  (test-case "grad shares storage: zero-grad! resets accumulation"
    (define x (requires-grad! (tensor '(1.0 2.0))))
    (backward! (sum (mul x x)))
    (check-equal? (tensor->list (grad x)) '(2.0 4.0))
    (zero-grad! x)
    (check-equal? (tensor->list (grad x)) '(0.0 0.0))
    (backward! (sum (mul x x)))
    (check-equal? (tensor->list (grad x)) '(2.0 4.0)))

  (test-case "zero-grad! before any backward is a no-op"
    (define x (requires-grad! (tensor '(1.0 2.0))))
    (zero-grad! x))

  (test-case "has-grad? and maybe-grad track gradient accumulation"
    (define x (requires-grad! (tensor '(1.0 2.0))))
    (check-false (has-grad? x))
    (check-false (maybe-grad x))
    (backward! (sum (mul x x)))
    (check-true (has-grad? x))
    (check-equal? (tensor->list (maybe-grad x)) '(2.0 4.0)))

  (test-case "manual SGD step: p -= lr * grad under with-no-grad"
    (define p (requires-grad! (tensor '(1.0 2.0))))
    (backward! (sum (mul p p)))
    (with-no-grad (sub! p (grad p) 0.25))
    (check-equal? (tensor->list p) '(0.5 1.0))
    (check-true (requires-grad? p)))

  (test-case "in-place zero! and mul!"
    (define t (tensor '(1.0 2.0)))
    (mul! t 3)
    (check-equal? (tensor->list t) '(3.0 6.0))
    (zero! t)
    (check-equal? (tensor->list t) '(0.0 0.0))))
