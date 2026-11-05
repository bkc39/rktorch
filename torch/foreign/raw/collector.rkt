#lang racket/base

(require (only-in "../../runtime/main-thread.rkt" on-main-os-thread?)
         (only-in "../../runtime/reaper.rkt" drain-deadline drain-reapers! wake-reapers!)
         (only-in "cas.rkt" cas-add! cas-claim-when!)
         (only-in "race-points.rkt" race-point))

(provide call-as-the-collector
         collect-and-wait!
         drain-deadline
         drain-finalizers!
         finalizer-runs
         note-finalizer-run!
         on-main-os-thread?
         wake-after-collection!)

;; Bumped by every release of a dead handle, by a reaper or a drain.
(define runs (box 0))

(define (note-finalizer-run!)
  (cas-add! runs 1))

(define (finalizer-runs)
  (unbox runs))

;; --- one collection at a time ---

;; A second thread that finds a trigger due while the first is inside its
;; collection skips instead of collecting again. The claim names its thread,
;; because kill-thread runs no dynamic-wind exit: a claimant that has died
;; holds nothing.
(define holder (box #f))

(define (unclaimed? held)
  (define free? (or (not held) (thread-dead? held)))
  (race-point collector-claiming free?)
  free?)

(define (claim-collector!)
  (cas-claim-when! holder unclaimed? (current-thread) #:site 'collector))

(define (call-as-the-collector thunk)
  (when (claim-collector!)
    (race-point collector-claimed (current-thread))
    (dynamic-wind void
                  thunk
                  (lambda ()
                    (race-point collector-releasing (current-thread))
                    (set-box! holder #f)))))

;; --- a collection that has really finished ---

(define collection-latch (make-thread-cell (make-weak-box (box 0))))

(define (wake-after-collection!)
  (unless (weak-box-value (thread-cell-ref collection-latch))
    (thread-cell-set! collection-latch (make-weak-box (box 0)))
    (wake-reapers!)))

(define (drain-finalizers!)
  (drain-reapers!))

(define (collect-and-wait!)
  (collect-garbage)
  (drain-finalizers!))
