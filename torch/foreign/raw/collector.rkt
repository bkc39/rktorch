#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic call-as-atomic))

(provide call-as-the-collector
         collect-and-wait!
         drain-deadline
         drain-finalizers!
         finalizer-runs
         note-finalizer-run!)

;; Bumped by every release, inside the ledger's atomic section.
(define runs 0)

(define (note-finalizer-run!)
  (set! runs (add1 runs)))

(define (finalizer-runs)
  runs)

;; --- one collection at a time ---

;; A second thread that finds a trigger due while the first is inside its
;; collection skips instead of collecting again. The claim names its thread,
;; because kill-thread runs no dynamic-wind exit: a claimant that has died
;; holds nothing.
(define holder #f)

(define (call-as-the-collector thunk)
  (define claimed?
    (call-as-atomic
     (lambda ()
       (and (or (not holder) (thread-dead? holder))
            (set! holder (current-thread))
            #t))))
  (when claimed?
    (dynamic-wind void
                  thunk
                  (lambda () (set! holder #f)))))

;; --- a collection that has really finished ---

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
