#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe/atomic in-atomic-mode?)
         (only-in racket/list filter-not)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "cas.rkt" cas-add! cas-take! cas-update!)
         (only-in "phantoms.rkt"
                  charge-epoch fresh-phantoms note-young! sync-phantoms! young-epoch?)
         (only-in "race-points.rkt" race-point)
         (only-in "regions.rkt" with-uninterruptible))

(provide fresh-counters
         counters-live counters-entries counters-since-check counters-since-sample
         note-counters-live!
         note-counters-mark!
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
  (provide counters-phantoms own-pending registered-tallies tally-count tally-limit tally-owner))

(define-syntax-parse-rule (tally-add! b:expr n:expr)
  (cas-add! b n #:site 'tally))

(define-syntax-parse-rule (tally-take! b:expr)
  (cas-take! b #:site 'tally))

(define-syntax-parse-rule (tally-update! b:expr f:expr)
  (cas-update! b f #:site 'tally))

(struct counters (index live entries since-check since-sample phantoms [limit #:mutable]))

(define counted (box 0))

(define (fresh-counters)
  (counters (sub1 (cas-add! counted 1)) (box 0) (box 0) (box 0) (box 0) (fresh-phantoms) #f))

(define (sync-counters-phantoms! c)
  (sync-phantoms! (counters-phantoms c) (counters-live c)))

(define (note-counters-live! c nbytes young)
  (cas-add! (counters-live c) nbytes)
  (note-young! (counters-phantoms c) young)
  (with-uninterruptible
    (sync-counters-phantoms! c)))

(struct pending (tally counters live entries since-check since-sample young young-epoch))

(define unheld (make-weak-box #f))

(struct tally (owner slots [ops #:mutable]))

(struct registry (count all))

(define flush-ops 64)
(define limit-cap (* 1024 1024))
(define limit-divisor 512)

(define (tally-limit mark)
  (if mark (min limit-cap (quotient mark limit-divisor)) limit-cap))

(define (limit-of c)
  (or (counters-limit c) 0))

(define (lower-limit! c limit)
  (set-counters-limit! c limit)
  (flush-tallies!))

(define (note-counters-mark! c mark)
  (define was (counters-limit c))
  (define limit (tally-limit mark))
  (cond
    [(not was) (set-counters-limit! c limit)]
    [(< limit was) (lower-limit! c limit)]
    [else (void)]))

(define (forget-mark! c)
  (when (counters-limit c)
    (lower-limit! c #f)))

(define current-tally (make-thread-cell #f #f))
(define tallies (box (registry 0 '())))

(define (registered-tallies)
  (registry-all (unbox tallies)))

(define (tally-count)
  (registry-count (unbox tallies)))

(define (own-tally)
  (or (thread-cell-ref current-tally)
      (let ([t (tally (make-weak-box (current-thread)) (box (vector)) 0)])
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
      (let ([p (pending t c (box 0) (box 0) (box 0) (box 0) (box 0) (box unheld))])
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

(define (own-pending c)
  (define p (own-slot c))
  (if p
      (map unbox (list (pending-live p) (pending-entries p)
                       (pending-since-check p) (pending-since-sample p)))
      '(0 0 0 0)))

(define (tally-staleness c)
  (* 2 (tally-count) (limit-of c)))

(define (tally-live! p nbytes entries young [epoch #f])
  (tally-add! (pending-live p) nbytes)
  (tally-add! (pending-entries p) entries)
  (unless (eqv? 0 young)
    (gather-young! p young epoch)))

(define (clamp n)
  (max 0 n))

(define (gather-young! p young epoch)
  (define held (pending-young-epoch p))
  (unless (or (eq? epoch (unbox held)) (young-epoch? (unbox held)))
    (tally-update! (pending-young p) clamp)
    (memory-order-release)
    (set-box! held (charge-epoch)))
  (tally-add! (pending-young p) young))

(define (tally-checked! p nbytes)
  (tally-add! (pending-since-check p) nbytes)
  (tally-add! (pending-since-sample p) nbytes))

(define (tally-op! p nbytes)
  (define t (pending-tally p))
  (define limit (limit-of (pending-counters p)))
  (define ops (add1 (tally-ops t)))
  (cond
    [(or (>= ops flush-ops)
         (>= nbytes limit)
         (>= (abs (unbox (pending-live p))) limit)
         (>= (unbox (pending-since-check p)) limit))
     (flush! t)]
    [else (set-tally-ops! t ops)]))

(define (take! from)
  (if (eqv? 0 (unbox from)) 0 (tally-take! from)))

(define (take-young! p)
  (define held (unbox (pending-young-epoch p)))
  (memory-order-acquire)
  (define n (take! (pending-young p)))
  (if (young-epoch? held) n (clamp n)))

(define (move! from to)
  (define n (take! from))
  (unless (eqv? 0 n)
    (cas-add! to n))
  n)

(define (flush! t #:sync-all? [sync-all? #f])
  (with-uninterruptible
    (for ([p (in-vector (unbox (tally-slots t)))] #:when p)
      (define c (pending-counters p))
      (define live (move! (pending-live p) (counters-live c)))
      (define young (take-young! p))
      (move! (pending-entries p) (counters-entries c))
      (move! (pending-since-check p) (counters-since-check c))
      (move! (pending-since-sample p) (counters-since-sample c))
      (note-young! (counters-phantoms c) young)
      (define quiet? (and (eqv? 0 live) (eqv? 0 young) (not sync-all?)))
      (unless quiet?
        (sync-counters-phantoms! c)))
    (when (eq? (weak-box-value (tally-owner t)) (current-thread))
      (set-tally-ops! t 0))))

(define (flush-tally!)
  (define t (thread-cell-ref current-tally))
  (when t (flush! t)))

(define (prunable? owner ask?)
  (define th (weak-box-value owner))
  (or (not th) (and ask? (thread-dead? th))))

(define (flush-tallies!)
  (define ask? (not (in-atomic-mode?)))
  (race-point tallies-flushing ask?)
  (define dead
    (for/fold ([dead '()]) ([t (in-list (registered-tallies))])
      (define gone? (prunable? (tally-owner t) ask?))
      (race-point tally-judged t)
      (flush! t #:sync-all? #t)
      (if gone? (cons t dead) dead)))
  (unless (null? dead)
    (tally-update! tallies
                   (lambda (r)
                     (define kept (filter-not (lambda (t) (memq t dead)) (registry-all r)))
                     (registry (length kept) kept))))
  (void))
