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

  (define (kill-inside-scope! nested?)
    (define entered (make-semaphore 0))
    (define (park) (ones 3) (semaphore-post entered) (sync never-evt))
    (define t (thread (lambda ()
                        (with-tensor-scope
                          (ones 3)
                          (if nested? (with-tensor-scope (park)) (park))))))
    (semaphore-wait entered)
    (kill-thread t))

  (define (found) (hash-ref (scope-diagnostics) 'scope-finalizer-found))

  (define (collect-until-found before)
    (let loop ([tries 0])
      (collect-garbage)
      (sleep 0.01)
      (when (and (= (found) before) (< tries 100))
        (loop (add1 tries)))))

  (for ([nested? '(#f #t)])
    (test-case (format "a killed thread's scopes are freed by a finalizer (nested ~a)" nested?)
      (define before (found))
      (kill-inside-scope! nested?)
      (collect-until-found before)
      (check-equal? (- (found) before) 1)))

  (test-case "tensor-keep moves a temporary out of the scope"
    (define kept #f)
    (with-tensor-scope
      (set! kept (tensor-keep (ones 4)))
      (void))
    (check-true (tensor? kept))
    (check-equal? (item (sum kept)) 4.0)))
