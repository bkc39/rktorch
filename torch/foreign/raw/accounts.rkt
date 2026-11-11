#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe/atomic
                  make-uninterruptible-lock
                  uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         (only-in racket/match match-define)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "cas.rkt" cas-add! cas-take! cas-update!)
         (only-in "pressure-settings.rkt" margin-over)
         (only-in "race-points.rkt" race-point)
         (only-in "tallies.rkt"
                  counters-entries counters-live counters-since-check
                  counters-since-sample flush-tallies! forget-mark!
                  fresh-counters own-pending-checks pending-of tally-checked! tally-live!
                  [note-mark! note-counters-mark!]))

(provide call-with-ledger
         with-ledger
         account-of
         account-capacity set-account-capacity!
         account-interval set-account-interval!
         account-release-after set-account-release-after!
         account-sample set-account-sample!
         account-trial set-account-trial!
         live-bytes since-check-bytes since-sample-bytes
         note-mark!
         own-checks
         tally-of
         count-live!
         count-checked!
         ledger-entries
         note-accounted!
         note-checked!
         note-live!
         note-unaccounted!
         checks-seen
         reset-checks!
         take-since-sample!
         live-bytes-by-device
         since-check-of
         unaccounted-bytes-by-device
         ledger-total
         unaccounted-total
         devices-over-floor
         pressure-reading
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

;; Nothing inside the lock blocks, sleeps or posts.
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

(define-syntax-parse-rule (with-ledger body:expr ...+)
  (call-with-ledger (lambda () body ...)))

(struct account
  (counters
   [sample #:mutable] [interval #:mutable] [capacity #:mutable]
   [floor #:mutable] shadow [unaccounted #:mutable] [release-after #:mutable]
   [trial #:mutable]))

(struct stats (backstop-collections reclaimed trough-collections trough-minors)
  #:mutable)

(module+ private
  (provide account-counters accounts raw-live))

(define accounts (box (hash)))
(define the-stats (stats 0 0 0 0))

(define (fresh-account)
  (account (fresh-counters) 0 #f 'unknown 0 (make-phantom-bytes 0) 0 0.0 #f))

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

(define (raw-live a) (unbox (counters-live (account-counters a))))
(define (live-of a) (max 0 (raw-live a)))
(define (since-check-bytes a) (unbox (counters-since-check (account-counters a))))
(define (since-sample-bytes a) (unbox (counters-since-sample (account-counters a))))

(define (live-bytes a)
  (flush-tallies!)
  (live-of a))

(define (note-mark! a mark)
  (note-counters-mark! (account-counters a) mark))

(define (own-checks a)
  (own-pending-checks (account-counters a)))

(define (tally-of a)
  (pending-of (account-counters a)))

(define (note-live! a nbytes)
  (cas-add! (counters-live (account-counters a)) nbytes))

(define (note-checked! a nbytes)
  (define c (account-counters a))
  (cas-add! (counters-since-check c) nbytes)
  (cas-add! (counters-since-sample c) nbytes))

(define (note-accounted! a nbytes)
  (note-live! a nbytes)
  (note-checked! a nbytes))

(define (note-unaccounted! a nbytes)
  (note-live! a (- nbytes)))

;; `p` is the caller's tally, or #f for the paths that write the shared
;; counters under the ledger.
(define (count-live! a p nbytes entries)
  (cond
    [p (tally-live! p nbytes entries)]
    [else
     (note-live! a nbytes)
     (cas-add! (counters-entries (account-counters a)) entries)]))

(define (count-checked! a p nbytes)
  (if p (tally-checked! p nbytes) (note-checked! a nbytes)))

(define (ledger-entries)
  (flush-tallies!)
  (for/sum ([a (in-list (all-accounts))])
    (unbox (counters-entries (account-counters a)))))

(define (checks-seen)
  (flush-tallies!)
  (for/list ([a (in-list (all-accounts))])
    (cons a (since-check-bytes a))))

(define (reset-checks! seen)
  (for ([entry (in-list seen)])
    (match-define (cons a bytes) entry)
    (cas-update! (counters-since-check (account-counters a)) (lambda (now) (max 0 (- now bytes))))))

(define (take-since-sample! a)
  (cas-take! (counters-since-sample (account-counters a))))

(define (account-over-floor? a)
  (define floor (account-floor a))
  (> (- (live-of a) floor) (margin-over floor)))

(define (live-bytes-by-device)
  (flush-tallies!)
  (for/list ([(dev a) (in-hash (unbox accounts))])
    (cons dev (raw-live a))))

(define (since-check-of dev)
  (flush-tallies!)
  (since-check-bytes (account-of dev)))

(define (unaccounted-bytes-by-device)
  (flush-tallies!)
  (for/list ([(dev a) (in-hash (unbox accounts))])
    (cons dev (account-unaccounted a))))

(define (ledger-total)
  (flush-tallies!)
  (for/sum ([a (in-list (all-accounts))]) (live-of a)))

(define (unaccounted-total)
  (for/sum ([a (in-list (all-accounts))]) (account-unaccounted a)))

(define (devices-over-floor)
  (flush-tallies!)
  (for/list ([(dev a) (in-hash (unbox accounts))] #:when (account-over-floor? a))
    dev))

(define (pressure-reading dev)
  (define a (account-of dev))
  (max (live-bytes a) (account-sample a)))

(define (backstop-interval dev)
  (account-interval (account-of dev)))

(define (charge-shadow! a bytes)
  (set-account-unaccounted! a bytes)
  (set-phantom-bytes! (account-shadow a) bytes))

(define (set-shadow! a reading)
  (charge-shadow! a (max 0 (- reading (live-of a)))))

(define (note-adopted! a nbytes)
  (define taken (min nbytes (account-unaccounted a)))
  (charge-shadow! a (- (account-unaccounted a) taken))
  taken)

(define (note-unadopted! a nbytes)
  (charge-shadow! a (+ (account-unaccounted a) nbytes)))

;; `live` was read with `reading`: a ledger that has moved since describes
;; another moment, and lowers nothing.
(define (lower-shadow! a reading live)
  (define excess (max 0 (- reading live)))
  (when (and (= live (live-of a))
             (< excess (account-unaccounted a)))
    (charge-shadow! a excess)))

(define generation 0)
(define (shadow-generation) generation)
(define (next-shadow-generation!) (set! generation (add1 generation)))

(define (lower-floors!)
  (flush-tallies!)
  (with-ledger
    (for ([a (in-list (all-accounts))])
      (set-account-floor! a (min (account-floor a) (live-of a))))))

(define (settle-floors!)
  (for ([a (in-list (all-accounts))])
    (set-account-floor! a (live-of a))))

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
  (flush-tallies!)
  (with-ledger
    (for ([a (in-list (all-accounts))])
      (define c (account-counters a))
      (forget-mark! c)
      (set-box! (counters-since-check c) 0)
      (set-box! (counters-since-sample c) 0)
      (set-account-sample! a 0)
      (set-account-interval! a #f)
      (set-account-capacity! a 'unknown)
      (set-account-floor! a 0)
      (set-shadow! a 0)
      (set-account-release-after! a 0.0)
      (set-account-trial! a #f))
    (next-shadow-generation!)))
