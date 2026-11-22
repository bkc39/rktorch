#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt" + cpu-device item native-memory-use ones sum tensor?)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/raw/memory.rkt"
                    collect-and-drain! native-memory-use/fold tr-tensor-free/checked)
           (only-in (submod "../foreign/raw/memory.rkt" scope-private) registered-lifetime?)
           (only-in "../foreign/raw/scope.rkt" scope-diagnostics)
           (only-in "../foreign/scope.rkt" tensor-keep with-tensor-scope)
           (only-in "../foreign/structs.rkt" tensor-handle))

  (define (counter k) (hash-ref (scope-diagnostics) k))
  (define (exit-frees) (counter 'exit-frees))

  (define (cpu-bytes [use native-memory-use])
    (cond [(assoc (cpu-device) (use)) => cdr]
          [else 0]))

  ;; garbage of earlier cases is reaped first, so a delta here is this case's
  (define (settled-cpu-bytes)
    (collect-and-drain!)
    (cpu-bytes))

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

  (test-case "a scoped tensor has no lifetime entry; an escaped one is registered at the root"
    (define inside #f)
    (define kept
      (with-tensor-scope
        (define t (ones 8))
        (set! inside (registered-lifetime? (tensor-handle t)))
        t))
    (check-false inside)
    (check-true (registered-lifetime? (tensor-handle kept))))

  (test-case "the ledger charges a scoped tensor and the exit uncharges it"
    (define before (settled-cpu-bytes))
    (define during #f)
    (define during/fold #f)
    (with-tensor-scope
      (ones 1024)
      (set! during (cpu-bytes))
      (set! during/fold (cpu-bytes native-memory-use/fold))
      (void))
    (check-equal? (- during before) 4096)
    (check-equal? during/fold during "the fold sees the scope's table")
    (check-equal? (cpu-bytes) before))

  (test-case "an escaped tensor keeps its charge until it is freed"
    (define before (settled-cpu-bytes))
    (define kept (with-tensor-scope (ones 1024)))
    (check-equal? (- (cpu-bytes) before) 4096)
    (tensor-free! kept)
    (check-equal? (cpu-bytes) before))

  (test-case "a tensor freed inside its scope is not freed again at the exit"
    (define before (settled-cpu-bytes))
    (define frees (exit-frees))
    (with-tensor-scope
      (define t (ones 1024))
      (ones 4)
      (tensor-free! t)
      (void))
    (check-equal? (- (exit-frees) frees) 1)
    (check-equal? (cpu-bytes) before))

  (test-case "a handle the exit freed is not released again by a raw free"
    (define h #f)
    (with-tensor-scope
      (set! h (tensor-handle (ones 4)))
      (void))
    (check-not-exn (lambda () (tr-tensor-free/checked h))))

  (test-case "a nested scope's result is owned by its parent"
    (define inner #f)
    (define inner-registered? #t)
    (with-tensor-scope
      (set! inner (with-tensor-scope (ones 3)))
      (set! inner-registered? (registered-lifetime? (tensor-handle inner)))
      (check-true (tensor? inner))
      (void))
    (check-false inner-registered?)
    (check-false (tensor? inner)))

  (test-case "a non-local exit frees everything the scope owns"
    (define made #f)
    (define before (settled-cpu-bytes))
    (check-exn #rx"boom"
               (lambda ()
                 (with-tensor-scope
                   (set! made (ones 2))
                   (error "boom"))))
    (check-false (tensor? made))
    (check-equal? (cpu-bytes) before))

  (define (kill-inside-scope! nested?)
    (define entered (make-semaphore 0))
    (define (park) (ones 3) (semaphore-post entered) (sync never-evt))
    (define t (thread (lambda ()
                        (with-tensor-scope
                          (ones 3)
                          (if nested? (with-tensor-scope (park)) (park))))))
    (semaphore-wait entered)
    (kill-thread t))

  (define (found) (counter 'scope-finalizer-found))

  (define (collect-until-found before)
    (let loop ([tries 0])
      (collect-garbage)
      (sleep 0.01)
      (when (and (= (found) before) (< tries 200))
        (loop (add1 tries)))))

  (for ([nested? '(#f #t)])
    (test-case (format "a killed thread's scopes are freed by the reaper (nested ~a)" nested?)
      (collect-and-drain!)
      (define before (found))
      (define bytes-before (cpu-bytes))
      (kill-inside-scope! nested?)
      (collect-until-found before)
      (check-equal? (- (found) before) 1)
      (collect-and-drain!)
      (check-equal? (cpu-bytes) bytes-before)))

  (test-case "tensor-keep moves a temporary out of the scope"
    (define kept #f)
    (with-tensor-scope
      (set! kept (tensor-keep (ones 4)))
      (void))
    (check-true (tensor? kept))
    (check-true (registered-lifetime? (tensor-handle kept)))
    (check-equal? (item (sum kept)) 4.0)))
