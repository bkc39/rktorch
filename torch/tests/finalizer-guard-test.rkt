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
           (only-in "../foreign/raw/syntax.rkt" guard-finalizer))

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

  (test-case "guard-finalizer passes successful releases through"
    (define released '())
    (define guarded
      (guard-finalizer (lambda (t) (set! released (cons t released)))))
    (guarded 'a)
    (guarded 'b)
    (check-equal? released '(b a)))

  (test-case "guard-finalizer's predicate is exn:fail?, nothing broader"
    ;; a non-exn:fail raise (e.g. a raw symbol) passes through — the guard
    ;; is a targeted swallow, not a sink for arbitrary control flow.
    (define guarded (guard-finalizer (lambda (_) (raise 'not-an-exn))))
    (check-exn (lambda (v) (eq? v 'not-an-exn))
               (lambda () (guarded 'handle))))

  (test-case "(deallocator) cancels the pending finalizer — observably"
    ;; The mechanism tensor-free! relies on, pinned with a recording free
    ;; (a tensor-level version could not observe cancellation: a stale
    ;; finalizer's error would be swallowed by the very guard under test).
    (define freed '())
    (define (record-free p)
      (set! freed (cons p freed))
      (free p))
    (define alloc ((allocator record-free) (lambda () (malloc 16 'raw))))
    (define release ((deallocator) record-free))
    ;; Path 1 — dropped without explicit free: the finalizer fires once.
    (void (alloc))
    (collect-until (lambda () (pair? freed)))
    (check-equal? (length freed) 1 "finalizer ran for the dropped alloc")
    ;; Path 2 — explicit release, then drop: the (deallocator) wrap must
    ;; cancel the registration, so the count rises by exactly one (the
    ;; explicit call), and collection adds nothing.
    (let ([p (alloc)])
      (release p))
    (check-equal? (length freed) 2 "explicit release ran")
    (collect-until (lambda () (> (length freed) 2)) #:tries 10)
    (check-equal? (length freed) 2
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
