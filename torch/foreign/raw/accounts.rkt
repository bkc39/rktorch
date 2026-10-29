#lang racket/base

(require (only-in ffi/unsafe/atomic
                  make-uninterruptible-lock
                  uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         (only-in "cas.rkt" cas-add! cas-update!)
         (only-in "pressure-settings.rkt" margin-over)
         (only-in "race-points.rkt" race-point))

(provide call-with-ledger
         account-of
         account-capacity set-account-capacity!
         account-interval set-account-interval!
         account-release-after set-account-release-after!
         account-sample set-account-sample!
         account-trial set-account-trial!
         live-bytes since-check-bytes since-sample-bytes
         note-accounted!
         note-unaccounted!
         checks-seen
         reset-checks!
         unsample!
         live-bytes-by-device
         since-check-of
         unaccounted-bytes-by-device
         ledger-total
         unaccounted-total
         devices-over-floor
         pressure-reading
         live-of
         backstop-interval
         charge-shadow!
         set-shadow!
         note-adopted!
         note-unadopted!
         lower-shadow!
         shadow-generation
         next-shadow-generation!
         lower-floors!
         settle-floors!
         stats-trough-collections set-stats-trough-collections!
         stats-trough-minors set-stats-trough-minors!
         count-backstop-collection!
         note-reclaimed!
         the-stats
         pressure-diagnostics
         reset-accounts!)

;; An uninterruptible lock, not a semaphore or atomic mode: finalizers take
;; it in an uninterruptible region, where blocking is an internal error, and a parallel
;; thread takes it on its own OS thread. Nothing inside blocks, sleeps or
;; posts. The holder box makes it re-entrant; only the holder can see itself
;; there.
(define ledger-lock (make-uninterruptible-lock))
(define ledger-holder (box #f))

(define (call-with-ledger thunk)
  (cond
    [(eq? (unbox ledger-holder) (current-thread)) (thunk)]
    [else
     (dynamic-wind
      (lambda ()
        (uninterruptible-lock-acquire ledger-lock)
        (set-box! ledger-holder (current-thread)))
      thunk
      (lambda ()
        (set-box! ledger-holder #f)
        (uninterruptible-lock-release ledger-lock)))]))

;; One per device: what the ledger holds there and what the two triggers
;; remember about it. `live`, `since-check` and `since-sample` are boxes
;; every accounting updates by CAS; every other field is written only under
;; the ledger. `capacity` is 'unknown until queried, then bytes, #f for a
;; device that has none, or (retry-at . ms) after a failed query. `shadow`
;; charges the collector, as phantom bytes, for the `unaccounted` bytes the
;; allocator holds beyond the ledger, and is adjusted, never replaced.
;; `release-after` is when the backstop may next empty the device's cache,
;; and `trial`, while the last release awaits its credit, counts down the
;; bytes it must stay released for.
(struct account
  (live since-check since-sample
   [sample #:mutable] [interval #:mutable] [capacity #:mutable]
   [floor #:mutable] shadow [unaccounted #:mutable] [release-after #:mutable]
   [trial #:mutable]))

(struct stats (backstop-collections reclaimed trough-collections trough-minors)
  #:mutable)

(define accounts (box (hash)))
(define the-stats (stats 0 0 0 0))

(define (fresh-account)
  (account (box 0) (box 0) (box 0) 0 #f 'unknown 0 (make-phantom-bytes 0) 0 0.0 #f))

(define (all-accounts)
  (hash-values (unbox accounts)))

(define (account-of dev)
  (or (hash-ref (unbox accounts) dev #f)
      (let ([fresh (fresh-account)])
        (race-point account-created dev)
        (hash-ref (cas-update! accounts
                               (lambda (all)
                                 (if (hash-ref all dev #f) all (hash-set all dev fresh))))
                  dev))))

(define (live-bytes a) (unbox (account-live a)))
(define (since-check-bytes a) (unbox (account-since-check a)))
(define (since-sample-bytes a) (unbox (account-since-sample a)))

(define (note-accounted! a nbytes)
  (cas-add! (account-live a) nbytes)
  (cas-add! (account-since-check a) nbytes)
  (cas-add! (account-since-sample a) nbytes))

(define (note-unaccounted! a nbytes)
  (cas-update! (account-live a) (lambda (live) (max 0 (- live nbytes)))))

;; A reset subtracts what its collection saw, so bytes accounted while it
;; ran still count toward the next check.
(define (checks-seen)
  (for/list ([a (in-list (all-accounts))])
    (cons a (since-check-bytes a))))

(define (take-seen! b seen)
  (cas-update! b (lambda (now) (max 0 (- now seen)))))

(define (reset-checks! seen)
  (for ([entry (in-list seen)])
    (take-seen! (account-since-check (car entry)) (cdr entry))))

(define (unsample! a seen)
  (take-seen! (account-since-sample a) seen))

(define (account-reading a)
  (max (live-bytes a) (account-sample a)))

(define (account-over-floor? a)
  (define floor (account-floor a))
  (> (- (live-bytes a) floor) (margin-over floor)))

;; --- readings: plain reads ---

(define (live-bytes-by-device)
  (for/list ([(dev a) (in-hash (unbox accounts))])
    (cons dev (live-bytes a))))

(define (since-check-of dev)
  (since-check-bytes (account-of dev)))

(define (unaccounted-bytes-by-device)
  (for/list ([(dev a) (in-hash (unbox accounts))])
    (cons dev (account-unaccounted a))))

(define (ledger-total)
  (for/sum ([a (in-list (all-accounts))]) (live-bytes a)))

(define (unaccounted-total)
  (for/sum ([a (in-list (all-accounts))]) (account-unaccounted a)))

(define (devices-over-floor)
  (for/list ([(dev a) (in-hash (unbox accounts))] #:when (account-over-floor? a))
    dev))

(define (pressure-reading dev)
  (account-reading (account-of dev)))

(define (live-of dev)
  (live-bytes (account-of dev)))

(define (backstop-interval dev)
  (account-interval (account-of dev)))

;; --- the shadow, under the ledger ---

(define (charge-shadow! a bytes)
  (set-account-unaccounted! a bytes)
  (set-phantom-bytes! (account-shadow a) bytes))

(define (set-shadow! a reading)
  (charge-shadow! a (max 0 (- reading (live-bytes a)))))

;; A handle onto storage that existed before it, a gradient the backward
;; pass wrote, joins the ledger with bytes the shadow may already charge.
(define (note-adopted! a nbytes)
  (define taken (min nbytes (account-unaccounted a)))
  (charge-shadow! a (- (account-unaccounted a) taken))
  taken)

;; and leaves it when that handle is released, the storage still held
(define (note-unadopted! a nbytes)
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
  (when (and (= live (live-bytes a))
             (< excess (account-unaccounted a)))
    (charge-shadow! a excess)))

;; counts refreshes, so a gradient adopted since the last one is known
(define generation 0)
(define (shadow-generation) generation)
(define (next-shadow-generation!) (set! generation (add1 generation)))

;; --- floors and statistics, under the ledger ---

(define (lower-floors!)
  (call-with-ledger
   (lambda ()
     (for ([a (in-list (all-accounts))])
       (set-account-floor! a (min (account-floor a) (live-bytes a)))))))

(define (settle-floors!)
  (for ([a (in-list (all-accounts))])
    (set-account-floor! a (live-bytes a))))

(define (count-backstop-collection!)
  (set-stats-backstop-collections! the-stats (add1 (stats-backstop-collections the-stats))))

(define (note-reclaimed! bytes)
  (set-stats-reclaimed! the-stats (+ (stats-reclaimed the-stats) (max 0 bytes))))

(define (pressure-diagnostics)
  (list (cons 'pressure-collections (stats-backstop-collections the-stats))
        (cons 'pressure-reclaimed (stats-reclaimed the-stats))
        (cons 'trough-collections (stats-trough-collections the-stats))
        (cons 'trough-minors (stats-trough-minors the-stats))
        (cons 'trough-floor (for/sum ([a (in-list (all-accounts))])
                              (account-floor a)))))

(define (reset-accounts!)
  (call-with-ledger
   (lambda ()
     (for ([a (in-list (all-accounts))])
       (set-box! (account-since-check a) 0)
       (set-box! (account-since-sample a) 0)
       (set-account-sample! a 0)
       (set-account-interval! a #f)
       (set-account-capacity! a 'unknown)
       (set-account-floor! a 0)
       (set-shadow! a 0)
       (set-account-release-after! a 0.0)
       (set-account-trial! a #f))
     (next-shadow-generation!))))
