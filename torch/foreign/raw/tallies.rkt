#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe/atomic in-atomic-mode?)
         (only-in racket/list filter-not)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../runtime/main-thread.rkt" on-main-os-thread?)
         (only-in "cas.rkt" cas-add! cas-claim! cas-take! cas-update!)
         (only-in "groups.rkt" fresh-group group-add! group-used? settle!)
         (only-in "regions.rkt" with-uninterruptible))

(provide fresh-counters
         counters-live counters-entries counters-since-check counters-since-sample
         note-counters-mark!
         forget-mark!
         pending-of
         own-pending-checks
         tally-staleness
         tally-live!
         tally-checked!
         tally-op!
         tally-charge!
         tally-group-add!
         flush-tally!
         flush-tallies!)

(module+ private
  (provide own-group own-pending registered-tallies tally-count tally-limit tally-owner))

(define-syntax-parse-rule (tally-add! b:expr n:expr)
  (cas-add! b n #:site 'tally))

(define-syntax-parse-rule (tally-take! b:expr)
  (cas-take! b #:site 'tally))

(define-syntax-parse-rule (tally-update! b:expr f:expr)
  (cas-update! b f #:site 'tally))

(struct counters (index live entries since-check since-sample [limit #:mutable]))

(define counted (box 0))

(define (fresh-counters)
  (counters (sub1 (cas-add! counted 1)) (box 0) (box 0) (box 0) (box 0) 0))

(struct pending (tally counters live entries since-check since-sample group))

(struct tally (owner slots touched [ops #:mutable]))

(struct registry (count all))

(define flush-ops 64)
(define limit-cap (* 1024 1024))
(define limit-divisor 512)

(define (tally-limit mark)
  (if mark (min limit-cap (quotient mark limit-divisor)) limit-cap))

(define (set-limit! c limit)
  (define was (counters-limit c))
  (unless (eqv? limit was)
    (set-counters-limit! c limit)
    (when (< limit was)
      (flush-tallies!))))

(define (note-counters-mark! c mark)
  (set-limit! c (tally-limit mark)))

(define (forget-mark! c)
  (set-limit! c 0))

(define current-tally (make-thread-cell #f #f))
(define tallies (box (registry 0 '())))

(define (registered-tallies)
  (registry-all (unbox tallies)))

(define (tally-count)
  (registry-count (unbox tallies)))

(define (own-tally)
  (or (thread-cell-ref current-tally)
      (let ([t (tally (make-weak-box (current-thread)) (box (vector)) (box '()) 0)])
        (tally-update! tallies
                       (lambda (r) (registry (add1 (registry-count r)) (cons t (registry-all r)))))
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
      (let ([p (pending t c (box 0) (box 0) (box 0) (box 0) (box (fresh-group 0)))])
        (tally-update! (tally-slots t)
                       (lambda (slots) (with-slot slots (counters-index c) p)))
        p)))

(define (own-slot c)
  (define t (thread-cell-ref current-tally))
  (and t (slot-of t c)))

(define (own-pending-checks c)
  (define p (own-slot c))
  (if p
      (values (unbox (pending-since-check p)) (unbox (pending-since-sample p)))
      (values 0 0)))

(define (own-group c)
  (unbox (pending-group (pending-of c))))

(define (own-pending c)
  (define p (own-slot c))
  (if p
      (map unbox (list (pending-live p) (pending-entries p)
                       (pending-since-check p) (pending-since-sample p)))
      '(0 0 0 0)))

(define (tally-staleness c)
  (* 2 (tally-count) (counters-limit c)))

(define (tally-live! p nbytes entries)
  (tally-add! (pending-live p) nbytes)
  (tally-add! (pending-entries p) entries))

(define (tally-checked! p nbytes)
  (tally-add! (pending-since-check p) nbytes)
  (tally-add! (pending-since-sample p) nbytes))

(define (tally-group-add! p g nbytes)
  (when (group-add! g nbytes)
    (tally-update! (tally-touched (pending-tally p)) (lambda (gs) (cons g gs)))))

(define (tally-charge! p nbytes)
  (define g (unbox (pending-group p)))
  (tally-group-add! p g nbytes)
  g)

(define (tally-op! p nbytes)
  (define t (pending-tally p))
  (define limit (counters-limit (pending-counters p)))
  (define ops (add1 (tally-ops t)))
  (cond
    [(or (>= ops flush-ops)
         (>= nbytes limit)
         (>= (abs (unbox (pending-live p))) limit)
         (>= (unbox (pending-since-check p)) limit)
         (>= (unbox (pending-since-sample p)) limit))
     (flush! t)]
    [else (set-tally-ops! t ops)]))

(define (move! from to)
  (unless (eqv? 0 (unbox from))
    (cas-add! to (tally-take! from))))

(define (close! p)
  (define forming (pending-group p))
  (define g (unbox forming))
  (when (and (group-used? g)
             (cas-claim! forming g (fresh-group 0) #:site 'tally))
    (settle! g)))

(define (take-touched! t)
  (define touched (tally-touched t))
  (let retry ()
    (define gs (unbox touched))
    (if (or (null? gs) (cas-claim! touched gs '() #:site 'tally))
        gs
        (retry))))

(define (flush! t)
  (with-uninterruptible
    (for ([p (in-vector (unbox (tally-slots t)))] #:when p)
      (define c (pending-counters p))
      (move! (pending-live p) (counters-live c))
      (move! (pending-entries p) (counters-entries c))
      (move! (pending-since-check p) (counters-since-check c))
      (move! (pending-since-sample p) (counters-since-sample c))
      (close! p))
    (for-each settle! (take-touched! t))
    (when (eq? (weak-box-value (tally-owner t)) (current-thread))
      (set-tally-ops! t 0))))

(define (flush-tally!)
  (define t (thread-cell-ref current-tally))
  (when t (flush! t)))

(define (prunable? owner ask?)
  (define th (weak-box-value owner))
  (or (not th) (and ask? (thread-dead? th))))

(define (flush-tallies!)
  (define ask? (and (on-main-os-thread?) (not (in-atomic-mode?))))
  (define dead
    (for/fold ([dead '()]) ([t (in-list (registered-tallies))])
      (define gone? (prunable? (tally-owner t) ask?))
      (flush! t)
      (if gone? (cons t dead) dead)))
  (unless (null? dead)
    (tally-update! tallies
                   (lambda (r)
                     (define kept (filter-not (lambda (t) (memq t dead)) (registry-all r)))
                     (registry (length kept) kept))))
  (void))
