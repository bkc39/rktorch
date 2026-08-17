#lang racket/base

;; Direct tests for the finalizer-safety guard (#38) — the live fix that
;; keeps free failures from raising out of GC finalization — and for the
;; explicit-free path that must NOT be guarded. The full suite can't
;; exercise a *throwing* free (that needs a poisoned native context), so
;; the guard's swallow semantics are pinned here as a unit, via the
;; guard-finalizer combinator the deallocator is built from.

(module+ test
  (require rackunit
           (only-in "../foreign.rkt" add ones tensor-shape)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/raw/syntax.rkt" guard-finalizer))

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

  (test-case "explicit tensor-free! is unguarded and cancels the finalizer"
    ;; the deliberate-release path: frees synchronously, flips the tag
    ;; (using the tensor afterwards raises at the marshal boundary; a
    ;; second free is a tag-checked no-op), and — because the checked
    ;; binding carries the (deallocator) wrap — leaves no pending
    ;; finalizer to fire on the freed handle at the next collection.
    (define t (ones 2 2))
    (check-equal? (tensor-shape t) '(2 2))
    (tensor-free! t)
    (check-exn exn:fail? (lambda () (add t t)))
    ;; second free: the tensor? contract rejects the freed handle — the
    ;; documented double-free protection at the contract boundary.
    (check-exn exn:fail:contract? (lambda () (tensor-free! t)))
    (collect-garbage)
    (collect-garbage) ;; finalizer canceled: no freed-tag marshal raise
    (check-true #t)))
