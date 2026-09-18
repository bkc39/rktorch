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
         margin-over
         pressure-diagnostics
         reset-pressure-state!
         allocator-reading
         drain-deadline
         native-memory-limit
         trough-budget
         trough-margin)

;; Atomic mode, not a semaphore: finalizers run in atomic mode, where
;; blocking is an internal error. Every field below is written inside it.
(define (call-with-ledger thunk)
  (call-as-atomic thunk))

;; One per device: what the ledger holds there and what the two triggers
;; remember about it. `mark` is 'unknown until queried, then bytes, #f for a
;; device with no capacity, or (retry-at . ms) after a failed query.
(struct account
  (live since-check since-sample sample interval mark floor)
  #:mutable)

(struct stats
  (finalizer-runs backstop-collections reclaimed trough-collections
   trough-minors)
  #:mutable)

;; when each trough stage may next run, and the thread inside a collection
(struct schedule (next-full-ms next-minor-ms holder) #:mutable)

(struct queries (capacity allocated) #:mutable)

(define accounts (make-hash))
(define the-stats (stats 0 0 0 0 0))
(define the-schedule (schedule 0.0 0.0 #f))
(define the-queries (queries #f #f))

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

;; --- the accounts, all inside the atomic section ---

(define (account-of dev)
  (hash-ref! accounts dev (lambda () (account 0 0 0 0 #f 'unknown 0))))

(define (note-accounted! dev nbytes)
  (define a (account-of dev))
  (set-account-live! a (+ (account-live a) nbytes))
  (set-account-since-check! a (+ (account-since-check a) nbytes))
  (set-account-since-sample! a (+ (account-since-sample a) nbytes)))

(define (note-unaccounted! dev nbytes)
  (define a (account-of dev))
  (set-account-live! a (max 0 (- (account-live a) nbytes))))

(define (reset-checks!)
  (for ([a (in-hash-values accounts)])
    (set-account-since-check! a 0)))

(define (account-reading a)
  (max (account-live a) (account-sample a)))

(define (account-over-floor? a)
  (define floor (account-floor a))
  (> (- (account-live a) floor) (margin-over floor)))

;; --- readings taken outside it ---

(define (live-bytes-by-device)
  (call-with-ledger
   (lambda ()
     (for/list ([(dev a) (in-hash accounts)])
       (cons dev (account-live a))))))

(define (ledger-total)
  (call-with-ledger
   (lambda ()
     (for/sum ([a (in-hash-values accounts)]) (account-live a)))))

(define (devices-over-floor)
  (call-with-ledger
   (lambda ()
     (for/list ([(dev a) (in-hash accounts)] #:when (account-over-floor? a))
       dev))))

(define (pressure-reading dev)
  (call-with-ledger (lambda () (account-reading (account-of dev)))))

;; --- statistics ---

(define (note-finalizer-run!)
  (set-stats-finalizer-runs! the-stats (add1 (stats-finalizer-runs the-stats))))

(define (finalizer-runs)
  (stats-finalizer-runs the-stats))

(define (note-reclaimed! bytes)
  (set-stats-reclaimed! the-stats (+ (stats-reclaimed the-stats) (max 0 bytes))))

(define (pressure-diagnostics)
  (list (cons 'pressure-collections (stats-backstop-collections the-stats))
        (cons 'pressure-reclaimed (stats-reclaimed the-stats))
        (cons 'trough-collections (stats-trough-collections the-stats))
        (cons 'trough-minors (stats-trough-minors the-stats))))

(define (reset-pressure-state!)
  (call-with-ledger
   (lambda ()
     (for ([a (in-hash-values accounts)])
       (set-account-since-check! a 0)
       (set-account-since-sample! a 0)
       (set-account-sample! a 0)
       (set-account-interval! a #f)
       (set-account-mark! a 'unknown)
       (set-account-floor! a 0))
     (set-schedule-next-full-ms! the-schedule 0.0)
     (set-schedule-next-minor-ms! the-schedule 0.0))))

;; --- the device's capacity and allocator, from raw/device.rkt ---

;; raw/device.rkt owns the two CUDA bindings and requires the ledger, so it
;; hands them over at instantiation instead of being required from here.
(define (install-cuda-queries! #:capacity capacity #:allocated allocated)
  (set-queries-capacity! the-queries capacity)
  (set-queries-allocated! the-queries allocated))

(define (cuda-query query dev)
  (and (eq? (device-type dev) 'cuda)
       query
       (query (device-index dev))))

(define (capacity-mark dev)
  (define total (cuda-query (queries-capacity the-queries) dev))
  (and total (positive? total) (floor (* high-water-fraction total))))

;; A mark is cached for good; a failed query on a CUDA device only until the
;; retry time, so one early failure cannot switch the backstop off.
(define (device-high-water dev)
  (or (native-memory-limit)
      (let ([cached (call-with-ledger
                     (lambda () (account-mark (account-of dev))))]
            [now (current-inexact-milliseconds)])
        (cond
          [(or (eq? cached 'unknown) (and (pair? cached) (>= now (cdr cached))))
           (define mark (capacity-mark dev))
           (call-with-ledger
            (lambda ()
              (set-account-mark! (account-of dev)
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
  (make-parameter
   (lambda (dev) (cuda-query (queries-allocated the-queries) dev))))

(define (sample-allocator! dev)
  (define reading (or ((allocator-reading) dev) 0))
  (call-with-ledger
   (lambda ()
     (define a (account-of dev))
     (set-account-sample! a reading)
     (set-account-since-sample! a 0))))

;; --- one collection at a time ---

;; A second thread that finds a trigger due while the first is inside its
;; collection skips instead of collecting again. The claim names its thread,
;; because kill-thread runs no dynamic-wind exit: a claimant that has died
;; holds nothing.
(define (call-as-the-collector thunk)
  (define claimed?
    (call-with-ledger
     (lambda ()
       (define holder (schedule-holder the-schedule))
       (and (or (not holder) (thread-dead? holder))
            (set-schedule-holder! the-schedule (current-thread))
            #t))))
  (when claimed?
    (dynamic-wind void
                  thunk
                  (lambda () (set-schedule-holder! the-schedule #f)))))

;; --- the backstop, from every accounting ---

(define (collect-under-pressure! dev)
  (unless (in-atomic-mode?)
    (define mark (device-high-water dev))
    (when mark
      (define base (quotient mark interval-divisor))
      (define-values (gate-open? sample-due?)
        (call-with-ledger
         (lambda ()
           (define a (account-of dev))
           (values (>= (account-since-check a) (or (account-interval a) base))
                   (>= (account-since-sample a) (quotient mark sample-divisor))))))
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
     (define a (account-of dev))
     (define reclaimed (max 0 (- before after)))
     (define interval (or (account-interval a) base))
     ;; A drain that ran out of time measured nothing, so it neither backs
     ;; the interval off nor counts as this device's check: the next
     ;; allocation looks again. Looking is cheap, and the mark still guards
     ;; the collection itself.
     (when drained?
       (set-account-interval! a
                              (if (< reclaimed (* reclaim-fraction mark))
                                  (min (* 2 interval) (* 2 mark))
                                  base))
       (reset-checks!))
     (set-stats-backstop-collections!
      the-stats (add1 (stats-backstop-collections the-stats)))
     (note-reclaimed! reclaimed))))

;; --- the troughs ---

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

(define (lower-floors!)
  (call-with-ledger
   (lambda ()
     (for ([a (in-hash-values accounts)])
       (set-account-floor! a (min (account-floor a) (account-live a)))))))

(define (settle-floors!)
  (for ([a (in-hash-values accounts)])
    (set-account-floor! a (account-live a))))

(define (due? next-ms)
  (and (>= (current-inexact-milliseconds) next-ms)
       (pair? (devices-over-floor))))

;; runs one stage: collect, then record its cost, its yield and its next
;; time. `collect!` reports whether its drain finished, and `on-drained!` is
;; what only a finished one earns: a floor pinned to bytes still waiting to
;; be freed would hide that residue from the next trough. The next time is
;; set either way, so a stalled stage cannot spin.
(define (trough-stage! next-ms set-next! bump! collect! on-drained! budget)
  (when (due? (next-ms the-schedule))
    (define started (current-inexact-milliseconds))
    (define before (ledger-total))
    (define drained? (collect!))
    (define finished (current-inexact-milliseconds))
    (define after (ledger-total))
    (call-with-ledger
     (lambda ()
       (when drained?
         (on-drained!))
       (set-next! the-schedule (+ finished (/ (- finished started) budget)))
       (bump! the-stats)
       (note-reclaimed! (- before after))))))

(define (collect-young-at-trough! budget)
  (trough-stage! schedule-next-minor-ms
                 set-schedule-next-minor-ms!
                 (lambda (s) (set-stats-trough-minors! s (add1 (stats-trough-minors s))))
                 (lambda ()
                   (collect-garbage 'minor)
                   (drain-finalizers!))
                 void
                 budget))

(define (collect-old-at-trough! budget)
  (trough-stage! schedule-next-full-ms
                 set-schedule-next-full-ms!
                 (lambda (s)
                   (set-stats-trough-collections! s (add1 (stats-trough-collections s))))
                 (lambda ()
                   (define-values (_observed drained?) (collect-and-wait!))
                   drained?)
                 (lambda ()
                   (settle-floors!)
                   (reset-checks!))
                 budget))

(define (collect-at-trough! #:young? [young? #f])
  (unless (in-atomic-mode?)
    (lower-floors!)
    (define budget (if young? (/ (trough-budget) 2) (trough-budget)))
    (call-as-the-collector
     (lambda ()
       (when young?
         (collect-young-at-trough! budget))
       (collect-old-at-trough! budget)))))

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
