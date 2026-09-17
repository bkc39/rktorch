#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic call-as-atomic in-atomic-mode?)
         (only-in "../device-type.rkt" device-index device-type))

(provide call-with-ledger
         call-as-the-collector
         note-accounted!
         note-unaccounted!
         note-finalizer-run!
         finalizer-runs
         live-bytes-by-device
         install-cuda-queries!
         collect-under-pressure!
         collect-at-trough!
         collect-and-wait!
         pressure-diagnostics
         reset-pressure-state!
         allocator-reading
         native-memory-limit
         trough-budget
         trough-margin)

;; Atomic mode, not a semaphore: finalizers run in atomic mode, where
;; blocking is an internal error.
(define (call-with-ledger thunk)
  (call-as-atomic thunk))

(define live-bytes (make-hash))
(define accounted-since (make-hash))
(define sampled-since (make-hash))
(define allocator-sample (make-hash))
(define collect-interval (make-hash))
(define high-water (make-hash))
(define trough-floor (make-hash))

(define finalizer-run-count (box 0))
(define pressure-collection-count (box 0))
(define pressure-reclaimed-bytes (box 0))
(define trough-collection-count (box 0))
(define trough-minor-count (box 0))
(define next-trough-ms (box 0.0))
(define next-minor-ms (box 0.0))
(define collector (box #f))

(define native-memory-limit (make-parameter #f))
(define high-water-fraction 4/5)
(define interval-divisor 8)
(define sample-divisor 32)
(define reclaim-fraction 1/20)
(define capacity-retry-ms 1000.0)

;; #f: the floor itself, kept between the two bounds below
(define trough-margin (make-parameter #f))
(define trough-margin-min (* 256 1024 1024))
(define trough-margin-max (* 1024 1024 1024))

;; the share of wall-clock time trough collections may take
(define trough-budget (make-parameter 1/20))

;; The three below run inside the caller's call-with-ledger section.
(define (note-accounted! dev nbytes)
  (hash-update! live-bytes dev (lambda (n) (+ n nbytes)) 0)
  (hash-update! accounted-since dev (lambda (n) (+ n nbytes)) 0)
  (hash-update! sampled-since dev (lambda (n) (+ n nbytes)) 0))

(define (note-unaccounted! dev nbytes)
  (hash-update! live-bytes dev (lambda (n) (max 0 (- n nbytes))) 0))

(define (note-finalizer-run!)
  (set-box! finalizer-run-count (add1 (unbox finalizer-run-count))))

(define (finalizer-runs)
  (unbox finalizer-run-count))

(define (live-bytes-by-device)
  (call-with-ledger (lambda () (hash->list live-bytes))))

(define (pressure-diagnostics)
  (list (cons 'pressure-collections (unbox pressure-collection-count))
        (cons 'pressure-reclaimed (unbox pressure-reclaimed-bytes))
        (cons 'trough-collections (unbox trough-collection-count))
        (cons 'trough-minors (unbox trough-minor-count))))

;; raw/device.rkt owns the two CUDA bindings and requires the ledger, so it
;; hands them over at instantiation instead of being required from here.
(define cuda-capacity-query (box #f))
(define cuda-allocated-query (box #f))

(define (install-cuda-queries! #:capacity capacity #:allocated allocated)
  (set-box! cuda-capacity-query capacity)
  (set-box! cuda-allocated-query allocated))

(define (cuda-query query dev)
  (and (eq? (device-type dev) 'cuda)
       (unbox query)
       ((unbox query) (device-index dev))))

(define (capacity-mark dev)
  (define total (cuda-query cuda-capacity-query dev))
  (and total (positive? total) (floor (* high-water-fraction total))))

;; A mark is cached for good; a failed query on a CUDA device only until the
;; retry time, so one early failure cannot switch the backstop off.
(define (device-high-water dev)
  (or (native-memory-limit)
      (let ([cached (call-with-ledger
                     (lambda () (hash-ref high-water dev 'unknown)))]
            [now (current-inexact-milliseconds)])
        (cond
          [(or (eq? cached 'unknown) (and (pair? cached) (>= now (cdr cached))))
           (define mark (capacity-mark dev))
           (call-with-ledger
            (lambda ()
              (hash-set! high-water dev
                         (cond
                           [mark mark]
                           [(eq? (device-type dev) 'cuda)
                            (cons 'retry-at (+ now capacity-retry-ms))]
                           [else #f]))))
           mark]
          [(pair? cached) #f]
          [else cached]))))

;; The allocator's own allocated bytes: the ledger double-counts views and
;; cannot see storage only the autograd graph holds. #f when unknown.
(define allocator-reading
  (make-parameter (lambda (dev) (cuda-query cuda-allocated-query dev))))

(define (sample-allocator! dev)
  (define reading (or ((allocator-reading) dev) 0))
  (call-with-ledger
   (lambda ()
     (hash-set! allocator-sample dev reading)
     (hash-set! sampled-since dev 0)))
  reading)

(define (pressure-reading dev)
  (call-with-ledger
   (lambda ()
     (max (hash-ref live-bytes dev 0)
          (hash-ref allocator-sample dev 0)))))

;; One collection at a time: a second thread that finds the gate open while
;; the first is inside its collection skips instead of collecting again. The
;; claim names its thread, because kill-thread runs no dynamic-wind exit: a
;; claimant that has died holds nothing.
(define (call-as-the-collector thunk)
  (define claimed?
    (call-with-ledger
     (lambda ()
       (define holder (unbox collector))
       (and (or (not holder) (thread-dead? holder))
            (set-box! collector (current-thread))
            #t))))
  (when claimed?
    (dynamic-wind void
                  thunk
                  (lambda () (set-box! collector #f)))))

(define (reset-accounted-since!)
  (for ([dev (in-list (hash-keys accounted-since))])
    (hash-set! accounted-since dev 0)))

(define (collect-under-pressure! dev)
  (unless (in-atomic-mode?)
    (define mark (device-high-water dev))
    (when mark
      (define base (quotient mark interval-divisor))
      (define-values (gate-open? sample-due?)
        (call-with-ledger
         (lambda ()
           (values (>= (hash-ref accounted-since dev 0)
                       (hash-ref collect-interval dev base))
                   (>= (hash-ref sampled-since dev 0)
                       (quotient mark sample-divisor))))))
      (when gate-open?
        (when sample-due?
          (sample-allocator! dev))
        (when (> (pressure-reading dev) mark)
          (call-as-the-collector
           (lambda () (pressure-collect! dev mark base))))))))

;; Reclaiming little means the working set itself sits above the mark, so
;; the interval to the next collection doubles instead of thrashing. A drain
;; that ran out of time says nothing about the working set.
(define (pressure-collect! dev mark base)
  (define before (pressure-reading dev))
  (define-values (_observed drained?) (collect-and-wait!))
  (sample-allocator! dev)
  (define after (pressure-reading dev))
  (call-with-ledger
   (lambda ()
     (define reclaimed (max 0 (- before after)))
     (define interval (hash-ref collect-interval dev base))
     (hash-set! collect-interval dev
                (cond
                  [(not drained?) interval]
                  [(< reclaimed (* reclaim-fraction mark))
                   (min (* 2 interval) (* 2 mark))]
                  [else base]))
     (reset-accounted-since!)
     (set-box! pressure-collection-count
               (add1 (unbox pressure-collection-count)))
     (set-box! pressure-reclaimed-bytes
               (+ reclaimed (unbox pressure-reclaimed-bytes))))))

(define (reset-pressure-state!)
  (call-with-ledger
   (lambda ()
     (for ([table (in-list (list accounted-since sampled-since allocator-sample
                                 collect-interval high-water trough-floor))])
       (hash-clear! table))
     (set-box! next-trough-ms 0.0)
     (set-box! next-minor-ms 0.0))))

;; The end of backward! is a step's trough: the graph is released, the
;; forward's intermediates are dead, and little is live, so a collection
;; here reclaims the most, promotes the least, and leaves Racket's own
;; schedule a low baseline. The floor is the ledger's size after the last
;; collection at a trough, zero before the first so that the first step's
;; garbage is not mistaken for it, and residue past the margin above it is
;; due one.
;; A step's dead intermediates have aged past the nursery by then, so only
;; a full collection reaches them, and the budget spaces those out: after
;; one that took t, the next waits t over the budget. A forward pass with
;; gradients off is the other case: its garbage is as young as garbage gets,
;; so that trough asks for a minor collection first and pays for a full one
;; only with what survives, the two stages splitting the budget.
(define (margin-over floor)
  (or (trough-margin)
      (max trough-margin-min (min floor trough-margin-max))))

(define (devices-over-floor)
  (call-with-ledger
   (lambda ()
     (for/list ([(dev live) (in-hash live-bytes)]
                #:when (let ([floor (hash-ref trough-floor dev 0)])
                         (> (- live floor) (margin-over floor))))
       dev))))

(define (lower-trough-floors!)
  (call-with-ledger
   (lambda ()
     (for ([(dev live) (in-hash live-bytes)]
           #:when (hash-has-key? trough-floor dev))
       (hash-update! trough-floor dev (lambda (floor) (min floor live)))))))

(define (ledger-total)
  (call-with-ledger
   (lambda ()
     (for/sum ([live (in-hash-values live-bytes)]) live))))

(define (due? next-ms)
  (and (>= (current-inexact-milliseconds) (unbox next-ms))
       (pair? (devices-over-floor))))

(define (record-trough! next-ms count started before budget)
  (define finished (current-inexact-milliseconds))
  (define after (ledger-total))
  (call-with-ledger
   (lambda ()
     (set-box! next-ms (+ finished (/ (- finished started) budget)))
     (set-box! count (add1 (unbox count)))
     (set-box! pressure-reclaimed-bytes
               (+ (max 0 (- before after)) (unbox pressure-reclaimed-bytes))))))

(define (collect-young-at-trough! budget)
  (when (due? next-minor-ms)
    (define started (current-inexact-milliseconds))
    (define before (ledger-total))
    (collect-garbage 'minor)
    (drain-finalizers!)
    (record-trough! next-minor-ms trough-minor-count started before budget)))

(define (collect-old-at-trough! budget)
  (when (due? next-trough-ms)
    (define started (current-inexact-milliseconds))
    (define before (ledger-total))
    (collect-and-wait!)
    (call-with-ledger
     (lambda ()
       (for ([(dev live) (in-hash live-bytes)])
         (hash-set! trough-floor dev live))
       (reset-accounted-since!)))
    (record-trough! next-trough-ms trough-collection-count
                    started before budget)))

(define (collect-at-trough! #:young? [young? #f])
  (unless (in-atomic-mode?)
    (lower-trough-floors!)
    (define budget (if young? (/ (trough-budget) 2) (trough-budget)))
    (call-as-the-collector
     (lambda ()
       (when young?
         (collect-young-at-trough! budget))
       (collect-old-at-trough! budget)))))

;; The canary shows the finalizer thread has started on this collection's
;; batch, not that it has finished: finalization order is unspecified and
;; the thread runs only while this one yields. So yield until the run count
;; has stood still for a few turns, within a deadline. Two results: whether
;; the canary was seen, and whether the drain finished inside the deadline.
(define drain-deadline-ms 2000)
(define quiet-turns 3)

(define (collect-and-wait!)
  (define canary-finalized (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post canary-finalized)))
  (collect-garbage)
  (define observed (sync/timeout 0.5 canary-finalized))
  (values (and observed #t) (drain-finalizers!)))

(define (drain-finalizers!)
  (define deadline (+ (current-inexact-milliseconds) drain-deadline-ms))
  (let loop ([runs (finalizer-runs)] [quiet 0])
    (sleep 0)
    (define now (finalizer-runs))
    (cond
      [(> (current-inexact-milliseconds) deadline) #f]
      [(not (= now runs)) (loop now 0)]
      [(< (add1 quiet) quiet-turns) (loop now (add1 quiet))]
      [else #t])))
