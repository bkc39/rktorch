#lang racket/base

;; Diagnostic (#556): tensor scopes. A tensor made inside a scope is owned by
;; it, with no `lifetimes` entry and no reaper registration; the scope frees
;; what it owns at its exit, except what its results reach.

(require (only-in "../../runtime/reaper.rkt" reap-when-dead! reaped)
         (only-in "cas.rkt" cas-add!)
         (only-in "collector.rkt" note-finalizer-run!)
         (only-in "memory.rkt"
                  adopt-scoped! current-tensor-scope enable-scoping! forget-open-table!
                  free-scoped! note-open-table! tensor-scope tensor-scope-guard
                  tensor-scope-owned tensor-scope-parent)
         (only-in "regions.rkt" with-uninterruptible)
         (only-in "tallies.rkt" flush-tally!))

(provide call-with-tensor-scope/handles
         scope-diagnostics
         scope-keep!)

(define counter-names
  '(scopes exit-frees escaped-to-parent escaped-to-reaper
    scope-finalizer-runs scope-finalizer-found))

(define counters
  (for/hasheq ([k (in-list counter-names)]) (values k (box 0))))

(define (count! k [n 1])
  (define b (hash-ref counters k))
  (cas-add! b n))

(define (scope-diagnostics)
  (for/hasheq ([(k b) (in-hash counters)]) (values k (unbox b))))

(define (escape! scope h)
  (define owned (tensor-scope-owned scope))
  (define lt (hash-ref owned h #f))
  (when lt
    (define parent (tensor-scope-parent scope))
    (with-uninterruptible
      (hash-remove! owned h)
      (if parent
          (hash-set! (tensor-scope-owned parent) h lt)
          (adopt-scoped! h lt)))
    (count! (if parent 'escaped-to-parent 'escaped-to-reaper))))

;; One guard per root scope holds the tables of the scopes open on its
;; thread. It points at no scope: the reaper's guardian is ordered, so a
;; guarded object reachable from itself would never be reaped.
(struct guard ([tables #:mutable]))

;; A table is touched only by its scope's thread, or here once that thread
;; cannot reach it. A free killed mid-walk leaves its claims to a dead thread,
;; which this free may take over.
(define (reap-guard! g)
  (with-uninterruptible
    (with-handlers ([(lambda (_) #t) void])
      (count! 'scope-finalizer-runs)
      (define freed
        (for/sum ([owned (in-list (guard-tables g))])
          (forget-open-table! owned)
          (free-scoped! owned 'finalizing)))
      (for ([_ (in-range freed)]) (note-finalizer-run!))
      (when (positive? freed)
        (count! 'scope-finalizer-found)))))

(define guard-reaped (reaped 'tensor-scope-guard reap-guard! #:after-batch flush-tally!))

(define (open-scope parent)
  (define owned (make-hasheq))
  (define g
    (cond
      [parent (tensor-scope-guard parent)]
      [else (define g (guard '()))
            (reap-when-dead! guard-reaped g)
            g]))
  (with-uninterruptible
    (set-guard-tables! g (cons owned (guard-tables g)))
    (note-open-table! owned))
  (tensor-scope parent owned g))

(define (close-scope! scope)
  (define g (tensor-scope-guard scope))
  (define owned (tensor-scope-owned scope))
  (count! 'exit-frees (free-scoped! owned (current-thread)))
  (with-uninterruptible
    (forget-open-table! owned)
    (set-guard-tables! g (cdr (guard-tables g)))))

(define (call-with-tensor-scope/handles thunk handles-of)
  (enable-scoping!)
  (define parent (thread-cell-ref current-tensor-scope))
  (define scope (open-scope parent))
  (count! 'scopes)
  (dynamic-wind
   (lambda () (thread-cell-set! current-tensor-scope scope))
   (lambda ()
     (define results (call-with-values thunk list))
     (for ([h (in-list (handles-of results))])
       (escape! scope h))
     (apply values results))
   (lambda ()
     (thread-cell-set! current-tensor-scope parent)
     (parameterize-break #f
       (close-scope! scope)))))

(define (scope-keep! h)
  (define scope (thread-cell-ref current-tensor-scope))
  (when scope (escape! scope h)))
