#lang racket/base

(require (only-in ffi/unsafe/atomic in-atomic-mode?)
         (only-in racket/list filter-not)
         (only-in "../../runtime/main-thread.rkt" on-main-os-thread?)
         (only-in "cas.rkt" cas-add! cas-take! cas-update!)
         (only-in "regions.rkt" with-uninterruptible))

(provide fresh-counters
         counters-live counters-entries counters-since-check counters-since-sample
         note-mark!
         forget-mark!
         pending-of
         own-pending-checks
         tally-staleness
         tally-live!
         tally-checked!
         tally-op!
         flush-tally!
         flush-tallies!)

(module+ private
  (provide own-pending tallies tally-limit tally-owner))

;; One device's shared ledger counters. Shared live is signed: a release
;; flushed before its charge takes it below zero, so readers clamp.
(struct counters (index live entries since-check since-sample [limit #:mutable]))

(define counted (box 0))

(define (fresh-counters)
  (counters (sub1 (cas-add! counted 1)) (box 0) (box 0) (box 0) (box 0) 0))

;; One thread's deltas against one device's counters, written by that thread
;; and taken by any flush.
(struct pending (tally counters live entries since-check since-sample))

(struct tally (owner slots [ops #:mutable]))

(define flush-ops 64)
(define limit-cap (* 1024 1024))
(define limit-divisor 512)

;; A device with no mark has no gate to open late, so only the cap applies.
(define (tally-limit mark)
  (if mark (min limit-cap (quotient mark limit-divisor)) limit-cap))

(define (note-mark! c mark)
  (define limit (tally-limit mark))
  (unless (eqv? limit (counters-limit c))
    (set-counters-limit! c limit)))

(define (forget-mark! c)
  (set-counters-limit! c 0))

(define current-tally (make-thread-cell #f #f))
(define tallies (box '()))

(define (own-tally)
  (or (thread-cell-ref current-tally)
      (let ([t (tally (make-weak-box (current-thread)) (box (vector)) 0)])
        (cas-update! tallies (lambda (all) (cons t all)) #:site 'tally)
        (thread-cell-set! current-tally t)
        t)))

(define (with-slot slots i p)
  (define grown (make-vector (max (vector-length slots) (add1 i)) #f))
  (vector-copy! grown 0 slots)
  (vector-set! grown i p)
  grown)

(define (slot-of t c)
  (define slots (unbox (tally-slots t)))
  (define i (counters-index c))
  (and (< i (vector-length slots)) (vector-ref slots i)))

(define (pending-of c)
  (define t (own-tally))
  (or (slot-of t c)
      (let ([p (pending t c (box 0) (box 0) (box 0) (box 0))])
        (cas-update! (tally-slots t)
                     (lambda (slots) (with-slot slots (counters-index c) p))
                     #:site 'tally)
        p)))

(define (own-pending-checks c)
  (define t (thread-cell-ref current-tally))
  (define p (and t (slot-of t c)))
  (if p
      (values (unbox (pending-since-check p)) (unbox (pending-since-sample p)))
      (values 0 0)))

(define (own-pending c)
  (define t (thread-cell-ref current-tally))
  (define p (and t (slot-of t c)))
  (if p
      (map unbox (list (pending-live p) (pending-entries p)
                       (pending-since-check p) (pending-since-sample p)))
      '(0 0 0 0)))

;; I4's bound: each tally holds back under the limit between its ops, and
;; at most one more tensor, under the limit too, inside one.
(define (tally-staleness c)
  (* 2 (length (unbox tallies)) (counters-limit c)))

(define (tally-live! p nbytes entries)
  (cas-add! (pending-live p) nbytes #:site 'tally)
  (cas-add! (pending-entries p) entries #:site 'tally))

(define (tally-checked! p nbytes)
  (cas-add! (pending-since-check p) nbytes #:site 'tally)
  (cas-add! (pending-since-sample p) nbytes #:site 'tally))

;; The caller's flush points T1 and T2: a limit of 0 flushes every op.
(define (tally-op! p nbytes)
  (define t (pending-tally p))
  (define limit (counters-limit (pending-counters p)))
  (define ops (add1 (tally-ops t)))
  (cond
    [(or (>= ops flush-ops)
         (>= nbytes limit)
         (>= (abs (unbox (pending-live p))) limit))
     (flush! t)]
    [else (set-tally-ops! t ops)]))

(define (move! from to)
  (unless (eqv? 0 (unbox from))
    (cas-add! to (cas-take! from #:site 'tally))))

;; A take and its add share one region, so a kill never strands the amount
;; between them.
(define (flush! t)
  (with-uninterruptible
    (for ([p (in-vector (unbox (tally-slots t)))] #:when p)
      (define c (pending-counters p))
      (move! (pending-live p) (counters-live c))
      (move! (pending-entries p) (counters-entries c))
      (move! (pending-since-check p) (counters-since-check c))
      (move! (pending-since-sample p) (counters-since-sample c)))
    (when (eq? (weak-box-value (tally-owner t)) (current-thread))
      (set-tally-ops! t 0))))

(define (flush-tally!)
  (define t (thread-cell-ref current-tally))
  (when t (flush! t)))

;; A dead or collected owner adds nothing more, so a tally found so before
;; its flush is empty after it and may go. `thread-dead?` enters atomic mode,
;; which suspends a parallel thread, and inside an uninterruptible region
;; such as the ledger's that is an internal error, so only the place's own
;; OS thread asks it, outside atomic mode.
(define (prunable? owner ask?)
  (define th (weak-box-value owner))
  (or (not th) (and ask? (thread-dead? th))))

(define (flush-tallies!)
  (define ask? (and (on-main-os-thread?) (not (in-atomic-mode?))))
  (define dead
    (for/fold ([dead '()]) ([t (in-list (unbox tallies))])
      (define gone? (prunable? (tally-owner t) ask?))
      (flush! t)
      (if gone? (cons t dead) dead)))
  (unless (null? dead)
    (cas-update! tallies
                 (lambda (all) (filter-not (lambda (t) (memq t dead)) all))
                 #:site 'tally))
  (void))
