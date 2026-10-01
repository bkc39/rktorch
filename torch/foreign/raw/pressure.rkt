#lang racket/base

(require (only-in ffi/unsafe/atomic call-as-atomic in-atomic-mode?)
         (only-in "../device-type.rkt" device-type)
         (only-in "collector.rkt"
                  call-as-the-collector collect-and-wait! drain-finalizers!)
         (only-in "pressure-settings.rkt"
                  margin-over native-collect-at-troughs native-collect-budget
                  native-collect-margin native-memory-fraction
                  native-memory-limit release-spacing))

(provide call-with-ledger
         note-accounted!
         note-adopted!
         note-unadopted!
         shadow-generation
         note-unaccounted!
         live-bytes-by-device
         unaccounted-bytes-by-device
         shadow-refresh
         refresh-shadows!
         lower-shadows!
         install-device-queries!
         backstop-interval
         collect-under-pressure!
         collect-at-trough!
         pressure-diagnostics
         reset-pressure-state!
         allocator-reading
         (all-from-out "pressure-settings.rkt"))

;; Atomic mode, not a semaphore: finalizers run in atomic mode, where
;; blocking is an internal error. Every field below is written inside it.
(define (call-with-ledger thunk)
  (call-as-atomic thunk))

;; One per device: what the ledger holds there and what the two triggers
;; remember about it. `capacity` is 'unknown until queried, then bytes, #f for
;; a device that has none, or (retry-at . ms) after a failed query. `shadow`
;; charges the collector, as phantom bytes, for the `unaccounted` bytes the
;; allocator holds beyond the ledger, and is adjusted, never replaced.
;; `release-after` is when the backstop may next empty the device's cache, and
;; `trial`, while the last release awaits its credit, counts down the bytes it
;; must stay released for.
(struct account
  ([live #:mutable] [since-check #:mutable] [since-sample #:mutable]
   [sample #:mutable] [interval #:mutable] [capacity #:mutable]
   [floor #:mutable] shadow [unaccounted #:mutable] [release-after #:mutable]
   [trial #:mutable]))

(struct stats (backstop-collections reclaimed trough-collections trough-minors)
  #:mutable)

;; when each trough stage may next run
(struct schedule (next-full-ms next-minor-ms) #:mutable)

(struct queries (capacity allocated release) #:mutable)

(define accounts (make-hash))
(define the-stats (stats 0 0 0 0))
(define the-schedule (schedule 0.0 0.0))
(define the-queries (queries #f #f #f))

(define interval-divisor 8)
(define sample-divisor 32)
(define reclaim-fraction 1/20)
(define capacity-retry-ms 1000.0)

;; --- the accounts, all inside the atomic section ---

(define (account-of dev)
  (hash-ref! accounts dev
             (lambda ()
               (account 0 0 0 0 #f 'unknown 0 (make-phantom-bytes 0) 0 0.0 #f))))

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

(define (unaccounted-bytes-by-device)
  (call-with-ledger
   (lambda ()
     (for/list ([(dev a) (in-hash accounts)])
       (cons dev (account-unaccounted a))))))

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

(define (note-reclaimed! bytes)
  (set-stats-reclaimed! the-stats (+ (stats-reclaimed the-stats) (max 0 bytes))))

(define (pressure-diagnostics)
  (list (cons 'pressure-collections (stats-backstop-collections the-stats))
        (cons 'pressure-reclaimed (stats-reclaimed the-stats))
        (cons 'trough-collections (stats-trough-collections the-stats))
        (cons 'trough-minors (stats-trough-minors the-stats))
        (cons 'trough-floor (for/sum ([a (in-hash-values accounts)])
                              (account-floor a)))))

(define (reset-pressure-state!)
  (call-with-ledger
   (lambda ()
     (for ([a (in-hash-values accounts)])
       (set-account-since-check! a 0)
       (set-account-since-sample! a 0)
       (set-account-sample! a 0)
       (set-account-interval! a #f)
       (set-account-capacity! a 'unknown)
       (set-account-floor! a 0)
       (set-shadow! a 0)
       (set-account-release-after! a 0.0)
       (set-account-trial! a #f))
     (next-shadow-generation!)
     (set-schedule-next-full-ms! the-schedule 0.0)
     (set-schedule-next-minor-ms! the-schedule 0.0))))

;; --- the device's capacity and allocator, from raw/device.rkt ---

;; raw/device.rkt owns the device bindings and requires the ledger, so it
;; hands these over at instantiation instead of being required from here.
;; Each takes a device and answers in bytes, or #f where it cannot say;
;; `release` empties the device's allocator cache and answers whether it had
;; one to empty.
(define (install-device-queries! #:capacity capacity
                                 #:allocated allocated
                                 #:release [release #f])
  (set-queries-capacity! the-queries capacity)
  (set-queries-allocated! the-queries allocated)
  (set-queries-release! the-queries release))

(define (queried-capacity dev)
  (define query (queries-capacity the-queries))
  (define total (and query (query dev)))
  (and total (positive? total) total))

;; The capacity is cached once known, and a failed query on a device that
;; should have one is retried after a moment, so one early failure cannot
;; switch the backstop off for the rest of the process.
(define (device-capacity dev)
  (define cached
    (call-with-ledger (lambda () (account-capacity (account-of dev)))))
  (define now (current-inexact-milliseconds))
  (cond
    [(or (eq? cached 'unknown) (and (pair? cached) (>= now (cdr cached))))
     (define total (queried-capacity dev))
     (call-with-ledger
      (lambda ()
        (set-account-capacity! (account-of dev)
                               (cond
                                 [total total]
                                 [(memq (device-type dev) '(cuda mps))
                                  (cons 'retry-at (+ now capacity-retry-ms))]
                                 [else #f]))))
     total]
    [(pair? cached) #f]
    [else cached]))

;; The fraction is applied at every check rather than folded into the cache,
;; so a program may parameterize it at any point in its run.
(define (device-high-water dev)
  (or (native-memory-limit)
      (let ([capacity (device-capacity dev)])
        (and capacity (floor (* (native-memory-fraction) capacity))))))

;; The allocator's own allocated bytes: the ledger double-counts views and
;; cannot see storage only the autograd graph holds. #f when unknown.
(define allocator-reading
  (make-parameter
   (lambda (dev)
     (define query (queries-allocated the-queries))
     (and query (query dev)))))

(define (live-of dev)
  (call-with-ledger (lambda () (account-live (account-of dev)))))

(define (backstop-interval dev)
  (call-with-ledger (lambda () (account-interval (account-of dev)))))

(define (sample-allocator! dev)
  (define live (live-of dev))
  (define known ((allocator-reading) dev))
  (define reading (or known 0))
  (call-with-ledger
   (lambda ()
     (define a (account-of dev))
     (define trial (account-trial a))
     (when trial
       (set-account-trial! a (- trial (account-since-sample a))))
     (set-account-sample! a reading)
     (set-account-since-sample! a 0)
     (when known
       (if (eq? (shadow-refresh) 'samples)
           (set-shadow! a reading)
           (lower-shadow! a reading live))))))

;; --- the shadow: what the allocator holds that the ledger cannot see ---

;; Storage only the autograd graph or libtorch itself holds reaches no
;; wrapper, so the ledger never charges it; the allocator's figure less the
;; ledger's is that remainder, and never counts a ledger byte twice. 'troughs
;; refreshes it only at a trough, before its collection, so the collection
;; that follows sets Racket's next major trigger with the shadow already in
;; it. 'samples also refreshes it at the backstop's samples, which in a large
;; forward means tracking the graph as it grows; #f charges nothing.
(define shadow-refresh (make-parameter 'troughs))

(define (charge-shadow! a bytes)
  (set-account-unaccounted! a bytes)
  (set-phantom-bytes! (account-shadow a) bytes))

(define (set-shadow! a reading)
  (charge-shadow! a (max 0 (- reading (account-live a)))))

;; A handle onto storage that existed before it, a gradient the backward
;; pass wrote, joins the ledger with bytes the shadow may already charge.
(define (note-adopted! dev nbytes)
  (define a (account-of dev))
  (define taken (min nbytes (account-unaccounted a)))
  (charge-shadow! a (- (account-unaccounted a) taken))
  taken)

;; and leaves it when that handle is released, the storage still held
(define (note-unadopted! dev nbytes)
  (define a (account-of dev))
  (charge-shadow! a (+ (account-unaccounted a) nbytes)))

;; Between troughs a sample may only lower the shadow. On MPS the trough
;; charged the allocator's cache, and new tensors that reuse it, or a release
;; that empties it, bring the reading down, so those bytes stop being charged
;; twice; what the graph holds mid-forward raises the reading and is not
;; charged. `live` is the ledger as it stood when the allocator was read; if
;; another thread has accounted or released a tensor since, the two no
;; longer describe the same moment and nothing is lowered.
(define (lower-shadow! a reading live)
  (define excess (max 0 (- reading live)))
  (when (and (= live (account-live a))
             (< excess (account-unaccounted a)))
    (charge-shadow! a excess)))

;; for a cache emptied outside the backstop: the OOM retry, or by hand
(define (lower-shadows!)
  (for ([dev (in-list (call-with-ledger (lambda () (hash-keys accounts))))])
    (define live (live-of dev))
    (define reading ((allocator-reading) dev))
    (when reading
      (call-with-ledger
       (lambda () (lower-shadow! (account-of dev) reading live))))))

;; An allocator that cannot answer right now leaves its device's charge as
;; it was; with charging off every charge goes to zero.
(define (refresh-shadows!)
  (define charging? (shadow-refresh))
  (for ([dev (in-list (call-with-ledger (lambda () (hash-keys accounts))))])
    (define reading (and charging? ((allocator-reading) dev)))
    (when (or reading (not charging?))
      (call-with-ledger
       (lambda () (set-shadow! (account-of dev) (or reading 0))))))
  (call-with-ledger next-shadow-generation!))

;; counts refreshes, so a gradient adopted since the last one is known
(define generation 0)
(define (shadow-generation) generation)
(define (next-shadow-generation!) (set! generation (add1 generation)))

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
        (if (> (pressure-reading dev) mark)
            (call-as-the-collector
             (lambda () (pressure-collect! dev mark base)))
            (when sample-due?
              (call-with-ledger (lambda () (end-trial! (account-of dev) base)))))))))

;; A release empties the cache, and below the working set the next steps take
;; those bytes straight back. So it earns the reset a collection earns only
;; when the backstop looks again after another mark's worth of allocation
;; and finds the reading still under the mark. Until then a reading back over
;; it means the release's bytes came back, and the interval backs off.
(define (end-trial! a base)
  (define trial (account-trial a))
  (when (and trial (<= trial 0))
    (set-account-interval! a base)
    (set-account-trial! a #f)))

;; Reclaiming little means the working set itself sits above the mark, so
;; the interval to the next collection doubles instead of thrashing. A drain
;; that ran out of time says nothing about the working set.
(define (pressure-collect! dev mark base)
  (define before (pressure-reading dev))
  (define-values (_observed drained?) (collect-and-wait!))
  ;; after a drain that ran out of time, finalizers still to run would
  ;; refill the cache behind the release and spend its spacing for nothing
  (define released (and drained? (release-cache! dev)))
  (sample-allocator! dev)
  (define after (pressure-reading dev))
  (call-with-ledger
   (lambda ()
     (define a (account-of dev))
     (define reclaimed (max 0 (- before after)))
     (define interval (or (account-interval a) base))
     (define backed-off (min (* 2 interval) (* 2 mark)))
     (define reclaimed-little? (< reclaimed (* reclaim-fraction mark)))
     ;; A drain that ran out of time measured nothing, so it neither backs
     ;; the interval off nor counts as this device's check: the next
     ;; allocation looks again. Looking is cheap, and the mark still guards
     ;; the collection itself. A release the spacing held back left the
     ;; cache in the reading, which says nothing of the working set, so the
     ;; interval stays where it was, unless the last release is on trial.
     (when drained?
       (set-account-interval! a
                              (cond
                                [(account-trial a) backed-off]
                                [(eq? released 'spaced) interval]
                                [reclaimed-little? backed-off]
                                [(eq? released 'released) interval]
                                [else base]))
       (when (and (eq? released 'released) (not reclaimed-little?))
         (set-account-trial! a mark))
       (reset-checks!))
     (set-stats-backstop-collections!
      the-stats (add1 (stats-backstop-collections the-stats)))
     (note-reclaimed! reclaimed))))

;; Where the allocator's reading counts its cache, as on unified memory, a
;; collection frees tensors into that cache and the reading does not move,
;; so the backstop also empties it. Spaced in time, not bytes: emptying is
;; quick, but the blocks the next steps need then come from the driver again.
;; Answers 'released, 'spaced when the spacing held it back, or #f.
(define (release-cache! dev)
  (define release (queries-release the-queries))
  (cond
    [(not release) #f]
    [(< (current-inexact-milliseconds)
        (call-with-ledger (lambda () (account-release-after (account-of dev)))))
     'spaced]
    [(release dev)
     (define next (+ (current-inexact-milliseconds) (release-spacing)))
     (call-with-ledger
      (lambda () (set-account-release-after! (account-of dev) next)))
     'released]
    [else #f]))

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
  (cond
    [(due? (next-ms the-schedule))
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
        (note-reclaimed! (- before after))))
     drained?]
    [else 'skipped]))

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
    (refresh-shadows!)
    (define budget
      (if young? (/ (native-collect-budget) 2) (native-collect-budget)))
    (call-as-the-collector
     (lambda ()
       (define young (if young? (collect-young-at-trough! budget) 'skipped))
       (define old (collect-old-at-trough! budget))
       ;; what the last stage that ran freed may include storage only a
       ;; dropped graph held, and on MPS it moves finalized tensors into the
       ;; cache; the charge moves with it rather than adding to it, and only
       ;; a finished drain shows what was freed
       (when (eq? #t (if (eq? old 'skipped) young old))
         (define before (unaccounted-total))
         (refresh-shadows!)
         (call-with-ledger
          (lambda () (note-reclaimed! (- before (unaccounted-total))))))))))

(define (unaccounted-total)
  (call-with-ledger
   (lambda ()
     (for/sum ([a (in-hash-values accounts)]) (account-unaccounted a)))))
