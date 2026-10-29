#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic in-atomic-mode?)
         (only-in ffi/unsafe/vm vm-eval)
         (only-in "cas.rkt" cas-add!)
         (only-in "race-points.rkt" race-point))

(provide call-as-the-collector
         collect-and-wait!
         drain-deadline
         drain-finalizers!
         finalizer-runs
         note-finalizer-run!
         on-main-os-thread?
         yield-to-finalizers!)

;; Bumped by every finalizer that releases a handle.
(define runs (box 0))

(define (note-finalizer-run!)
  (cas-add! runs 1))

(define (finalizer-runs)
  (unbox runs))

;; Racket CS numbers main's OS thread 0, and coroutine threads share it.
(define os-thread-id (vm-eval 'get-thread-id))

(define (on-main-os-thread?)
  (zero? (os-thread-id)))

;; --- one collection at a time ---

;; A second thread that finds a trigger due while the first is inside its
;; collection skips instead of collecting again. The claim names its thread,
;; because kill-thread runs no dynamic-wind exit: a claimant that has died
;; holds nothing.
(define holder (box #f))

(define (claim-collector!)
  (define held (unbox holder))
  (define free? (or (not held) (thread-dead? held)))
  (race-point collector-claiming free?)
  (cond
    [(not free?) #f]
    [(box-cas! holder held (current-thread)) #t]
    [(eq? (unbox holder) held) (claim-collector!)]
    [else #f]))

(define (call-as-the-collector thunk)
  (when (claim-collector!)
    (race-point collector-claimed (current-thread))
    (dynamic-wind void
                  thunk
                  (lambda ()
                    (race-point collector-releasing (current-thread))
                    (set-box! holder #f)))))

;; --- a collection that has really finished ---

(define (start-finalizer-thread!)
  (register-finalizer (box 0) void))

(start-finalizer-thread!)

;; A thread's latch dies in the first collection after it last yielded. Off
;; main's OS thread a `(sleep 0)` hops to main and gives the finalizer
;; thread nothing it would not get anyway.
(define collection-latch (make-thread-cell (make-weak-box (box 0))))

(define (yield-to-finalizers!)
  (cond
    [(weak-box-value (thread-cell-ref collection-latch)) (void)]
    ;; sleeping inside an atomic section is an error
    [(in-atomic-mode?) (void)]
    [(not (on-main-os-thread?)) (void)]
    [else
     (thread-cell-set! collection-latch (make-weak-box (box 0)))
     (sleep 0)]))

;; The canary shows the finalizer thread has started on this collection's
;; batch, not that it has finished: finalization order is unspecified and
;; the thread runs only while this one yields. So yield until the run count
;; has stood still for a few turns, within a deadline. Two results: whether
;; the canary was seen, and whether the drain finished inside the deadline.
(define drain-deadline (make-parameter 2000))
(define quiet-turns 3)

(define (collect-and-wait!)
  (define canary-finalized (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post canary-finalized)))
  (collect-garbage)
  (define observed (sync/timeout 0.5 canary-finalized))
  (values (and observed #t) (drain-finalizers!)))

(define (drain-finalizers!)
  (define deadline (+ (current-inexact-milliseconds) (drain-deadline)))
  (let loop ([runs (finalizer-runs)] [quiet 0])
    (sleep 0)
    (define now (finalizer-runs))
    (cond
      [(>= (current-inexact-milliseconds) deadline) #f]
      [(not (= now runs)) (loop now 0)]
      [(< (add1 quiet) quiet-turns) (loop now (add1 quiet))]
      [else #t])))
