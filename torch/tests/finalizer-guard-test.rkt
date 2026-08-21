#lang racket/base

;; Direct tests for the finalizer-safety guard (#38) — the live fix that
;; keeps free failures from raising out of GC finalization — and for the
;; explicit-free path that must NOT be guarded. The full suite can't
;; exercise a *throwing* free (that needs a poisoned native context), so
;; the guard's swallow semantics are pinned here as a unit, via the
;; guard-finalizer combinator the deallocator is built from.

(module+ test
  (require rackunit
           (only-in ffi/unsafe malloc free)
           (only-in ffi/unsafe/alloc allocator deallocator)
           (only-in "../foreign.rkt" add ones tensor-shape)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/raw/memory.rkt" guard-finalizer))

  ;; Finalizers run asynchronously after collection; poll with a bounded
  ;; wait so assertions are deterministic without being timing-fragile.
  (define (collect-until ready? #:tries [tries 50])
    (let loop ([i 0])
      (collect-garbage)
      (unless (or (ready?) (>= i tries))
        (sleep 0.01)
        (loop (add1 i)))))

  (test-case "guard-finalizer swallows exn:fail from the release"
    (define guarded
      (guard-finalizer (lambda (t) (error 'release "boom: ~a" t))))
    ;; must return normally — a raise here is the #38 cascade class
    (check-equal? (guarded 'handle) (void)))

  (test-case "guard-finalizer swallows are counted, not invisible (#51)"
    (define before (finalizer-failures))
    ((guard-finalizer (lambda (_t) (error 'boom "swallowed"))) 'handle)
    ((guard-finalizer (lambda (_t) (raise 'bare))) 'handle)
    (check-equal? (finalizer-failures) (+ before 2))
    ;; successful releases don't count
    ((guard-finalizer void) 'handle)
    (check-equal? (finalizer-failures) (+ before 2)))

  (test-case "guard-finalizer passes successful releases through"
    (define released '())
    (define guarded
      (guard-finalizer (lambda (t) (set! released (cons t released)))))
    (guarded 'a)
    (guarded 'b)
    (check-equal? released '(b a)))

  (test-case "guard-finalizer's catch is total — bare raises too"
    ;; ffi/unsafe/alloc requires a deallocate argument that NEVER raises;
    ;; a bare raised value escaping a finalizer re-enters the #38 cascade
    ;; just like an exn:fail would, so it must be swallowed as well.
    (define guarded (guard-finalizer (lambda (_) (raise 'not-an-exn))))
    (check-equal? (guarded 'handle) (void)))

  (test-case "(deallocator) cancels the pending finalizer — observably"
    ;; The mechanism tensor-free! relies on, mirrored in production
    ;; topology: the allocator side registers a guard-finalizer-wrapped
    ;; closure, the explicit side is a DIFFERENT (deallocator)-wrapped
    ;; closure (matching (allocator tr-tensor-free/finalizer) vs the
    ;; checked binding). Counters, not the pointers, are recorded — the
    ;; earlier version consed the pointer into a list, keeping it strongly
    ;; reachable so the negative assertion passed vacuously.
    (define finalizer-count 0)
    (define explicit-count 0)
    (define (finalizer-release p)
      (set! finalizer-count (add1 finalizer-count))
      (free p))
    (define (explicit-release p)
      (set! explicit-count (add1 explicit-count))
      (free p))
    (define alloc
      ((allocator (guard-finalizer finalizer-release))
       (lambda () (malloc 16 'raw))))
    (define release ((deallocator) explicit-release))
    ;; Path 1 — dropped without explicit free: the finalizer fires once.
    (void (alloc))
    (collect-until (lambda () (> finalizer-count 0)))
    (check-equal? finalizer-count 1 "finalizer ran for the dropped alloc")
    ;; Path 2 — explicit release through the distinct deallocator-wrapped
    ;; closure, then out of scope: cancellation must hold across the two
    ;; closures, so the finalizer count stays put after collections.
    (let ([p (alloc)])
      (release p))
    (check-equal? explicit-count 1 "explicit release ran")
    (collect-until (lambda () (> finalizer-count 1)) #:tries 10)
    (check-equal? finalizer-count 1
                  "no finalizer fired after explicit release — canceled"))

  (test-case "explicit tensor-free! is unguarded; freed handles are inert"
    ;; the deliberate-release path at tensor level: frees synchronously,
    ;; flips the tag (using the tensor afterwards raises at the marshal
    ;; boundary; a second free is rejected by the tensor? contract), and
    ;; collections after the free stay quiet — cancellation itself is
    ;; pinned observably by the mechanism test above.
    (define t (ones 2 2))
    (check-equal? (tensor-shape t) '(2 2))
    (tensor-free! t)
    (check-exn exn:fail? (lambda () (add t t)))
    (check-exn exn:fail:contract? (lambda () (tensor-free! t)))
    (collect-garbage)
    (collect-garbage)))
