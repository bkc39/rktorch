#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in "cas.rkt" cas-add!)
         (only-in "memory.rkt"
                  current-tensor-scope finalize-owned! free-owned!
                  set-tensor-scope-child! tensor-scope tensor-scope-child
                  tensor-scope-owned tensor-scope-parent)
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

;; A scope's table is touched only by its thread, or by its finalizer once
;; that thread cannot reach it. A free killed mid-walk leaves its claims to a
;; dead thread, which the finalizer may take over.
(define (free-all! scope)
  (define owned (tensor-scope-owned scope))
  (define freed
    (for/sum ([(h lt) (in-hash owned)])
      (if (free-owned! h lt) 1 0)))
  (hash-clear! owned)
  (define child (tensor-scope-child scope))
  (+ freed (if child (free-all! child) 0)))

(define (finalize-scope scope)
  (with-uninterruptible
    (with-handlers ([(lambda (_) #t) void])
      (count! 'scope-finalizer-runs)
      (when (positive? (free-all! scope))
        (count! 'scope-finalizer-found)))))

;; Only a root scope carries a finalizer: a nested one is reachable from its
;; parent's while it runs.
(define (call-with-tensor-scope/handles thunk handles-of)
  (define parent (thread-cell-ref current-tensor-scope))
  (define scope (tensor-scope parent (make-hasheq) #f))
  (count! 'scopes)
  (if parent
      (set-tensor-scope-child! parent scope)
      (register-finalizer scope finalize-scope))
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
       (count! 'exit-frees (free-all! scope))
       (when parent (set-tensor-scope-child! parent #f))))))

(define (scope-keep! h)
  (define scope (thread-cell-ref current-tensor-scope))
  (when scope (escape! scope h)))
