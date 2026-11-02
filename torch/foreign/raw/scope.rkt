#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in "cas.rkt" cas-add!)
         (only-in "collector.rkt" note-finalizer-run!)
         (only-in "memory.rkt"
                  current-tensor-scope finalize-owned! free-owned!
                  tensor-scope tensor-scope-guard tensor-scope-owned
                  tensor-scope-parent)
         (only-in "regions.rkt" with-uninterruptible))

(provide call-with-tensor-scope/handles
         scope-diagnostics
         scope-keep!)

(define counter-names
  '(scopes exit-frees escaped-to-parent escaped-to-finalizer
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
          (finalize-owned! h lt)))
    (count! (if parent 'escaped-to-parent 'escaped-to-finalizer))))

;; A table is touched only by its scope's thread, or by the guard's finalizer
;; once that thread cannot reach it. A free killed mid-walk leaves its claims
;; to a dead thread, which the finalizer may take over.
(define (free-table! owned [note-freed! void])
  (define freed
    (for/sum ([(h lt) (in-hash owned)])
      (cond
        [(free-owned! h lt) (note-freed!) 1]
        [else 0])))
  (hash-clear! owned)
  freed)

;; One guard per root scope holds the tables of the scopes open on its
;; thread. It points at no scope: finalization is ordered, so a guarded
;; object reachable from itself would never be finalized.
(struct guard ([tables #:mutable]))

(define (finalize-guard g)
  (with-uninterruptible
    (with-handlers ([(lambda (_) #t) void])
      (count! 'scope-finalizer-runs)
      (define freed
        (for/sum ([owned (in-list (guard-tables g))])
          (free-table! owned note-finalizer-run!)))
      (when (positive? freed)
        (count! 'scope-finalizer-found)))))

(define (open-scope parent)
  (define owned (make-hasheq))
  (define g
    (cond
      [parent (tensor-scope-guard parent)]
      [else (define g (guard '()))
            (register-finalizer g finalize-guard)
            g]))
  (with-uninterruptible
    (set-guard-tables! g (cons owned (guard-tables g))))
  (tensor-scope parent owned g))

(define (close-scope! scope)
  (define g (tensor-scope-guard scope))
  (count! 'exit-frees (free-table! (tensor-scope-owned scope)))
  (set-guard-tables! g (cdr (guard-tables g))))

(define (call-with-tensor-scope/handles thunk handles-of)
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
