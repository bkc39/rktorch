#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/dict dict-ref)
           (only-in racket/match match-define)
           rackunit
           (only-in "../foreign.rkt" cpu-device zeros)
           (only-in "../foreign/raw/accounts.rkt"
                    account-of charge-shadow! set-account-interval! set-account-sample!
                    take-since-sample! unaccounted-bytes-by-device with-ledger)
           (only-in (submod "../foreign/raw/accounts.rkt" private) account-counters)
           (only-in "../foreign/raw/cas.rkt" cas-claim!)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait!)
           (only-in "../foreign/raw/device-queries.rkt" allocator-reading)
           (only-in "../foreign/raw/memory.rkt"
                    native-memory-use/fold record-allocation! unaccount!)
           (only-in "../foreign/raw/pressure.rkt"
                    collect-under-pressure! lower-shadows! native-memory-limit
                    refresh-shadows!)
           (only-in "../foreign/raw/tallies.rkt" flush-tallies! flush-tally! pending-of)
           (only-in (submod "../foreign/raw/tallies.rkt" private)
                    tally-count tally-limit tally-owner)
           (only-in "../foreign/raw/regions.rkt" with-uninterruptible)
           (only-in "private/race-harness.rkt" join settle! spawn with-race-hook worker-thread)
           (only-in "private/tally-devices.rkt"
                    charging collections-of kib mib shared-live spin-until unaccount-all!
                    with-batched-device with-parked with-unmarked-device)
           (only-in "private/within.rkt" bounded-test))

  (define (fold-of dev) (dict-ref (native-memory-use/fold) dev 0))

  (bounded-test "a drain's #t waits for another thread's transfer of the drained releases"
    (settle!)
    (define cpu (cpu-device))
    (void (zeros 1))
    (define dropped (for/list ([_ (in-range 8)]) (zeros 16)))
    (flush-tally!)
    (define releaser (box #f))
    (define held (box 'waiting))
    (define flushed (box #f))
    (define resume (box #f))
    (define drained (box #f))
    (define flusher
      (spawn (lambda ()
               (spin-until (lambda () (unbox releaser)))
               (with-uninterruptible (flush-tallies!))
               (set-box! flushed #t))
             #:pool 'own))
    (define flusher-thread (worker-thread flusher))
    (define (hold-the-transfer name subject)
      (cond
        [(and (eq? name 'handle-released)
              (not (eq? (current-thread) flusher-thread))
              (cas-claim! releaser #f (current-thread)))
         (spin-until (lambda () (or (eq? (unbox held) 'parked) (unbox flushed))))]
        [(and (eq? name 'tally-moving)
              (eq? (current-thread) flusher-thread)
              (eq? (weak-box-value (tally-owner subject)) (unbox releaser))
              (cas-claim! held 'waiting 'parked))
         (spin-until (lambda () (unbox resume)))]
        [else (void)]))
    (set! dropped #f)
    (define early
      (with-race-hook hold-the-transfer
        (define drain
          (spawn (lambda ()
                   (define done? (collect-and-wait!))
                   (set-box! drained (list done? (- (shared-live cpu) (fold-of cpu)))))))
        (define watcher
          (spawn (lambda ()
                   (spin-until (lambda () (or (eq? (unbox held) 'parked) (unbox drained))))
                   (spin-until (lambda () (unbox drained)) #:seconds 1)
                   (begin0 (unbox drained)
                     (set-box! resume #t)))
                 #:pool 'own))
        (begin0 (join watcher)
          (join drain)
          (join flusher))))
    (check-eq? (unbox held) 'parked "another thread's flush held the releaser's transfer")
    (check-false early "the drain returned while the transfer was held")
    (match-define (list done? off) (unbox drained))
    (check-true done? "the drain finished")
    (check-equal? off 0 "the drained releases were in the shared counter at its #t"))

  (define (freed-during-reading dev key reading)
    (define freed (box #f))
    (lambda (d)
      (when (and (equal? d dev) (cas-claim! freed #f #t))
        (join (spawn (lambda () (unaccount! key)) #:pool 'own)))
      (and (equal? d dev) reading)))

  (define (unaccounted-of dev) (dict-ref (unaccounted-bytes-by-device) dev 0))

  (bounded-test "a free between a refresh's flush and its reading still charges the shadow"
    (with-batched-device (dev 65)
      (define key (box 'freed))
      (record-allocation! key (* 64 kib) dev)
      (flush-tally!)
      (parameterize ([allocator-reading (freed-during-reading dev key (* 32 kib))])
        (refresh-shadows!))
      (check-equal? (unaccounted-of dev) (* 32 kib)
                    "the allocator's 32 KiB less a ledger with the free in it")
      (parameterize ([allocator-reading (lambda (_) 0)])
        (refresh-shadows!))))

  (bounded-test "a free during a lowering's reading leaves the charge as it was"
    (with-batched-device (dev 66)
      (define a (account-of dev))
      (define key (box 'freed))
      (record-allocation! key (* 64 kib) dev)
      (flush-tally!)
      (with-ledger (charge-shadow! a (* 32 kib)))
      (parameterize ([allocator-reading (freed-during-reading dev key (* 32 kib))])
        (lower-shadows!))
      (check-equal? (unaccounted-of dev) (* 32 kib)
                    "the ledger moved during the reading, so it lowers nothing")
      (with-ledger (charge-shadow! a 0))))

  (bounded-test "a lowering killed before its flush leaves the gate seeing the crossing"
    (with-unmarked-device (dev 67)
      (define a (account-of dev))
      (define device-mark (* 10 mib))
      (define parked (box 'parked))
      (define crossing (box 'crossing))
      (parameterize ([native-memory-limit #f]
                     [allocator-reading (lambda (_) #f)])
        (collect-under-pressure! dev))
      (with-parked (_w (charging dev (list parked) (* 900 kib)))
        (void (pending-of (account-counters a)))
        (define gap (add1 (* 2 (tally-count) (tally-limit device-mark))))
        (check-true (< gap (* 900 kib))
                    (format "~a tallies leave a gap of ~a bytes, which the parked charge cannot cross"
                            (tally-count) gap))
        (define lowerer (box #f))
        (define killed (box #f))
        (define (die-in-the-flush name _subject)
          (when (and (eq? name 'tally-judged)
                     (eq? (current-thread) (unbox lowerer))
                     (cas-claim! killed #f #t))
            (kill-thread (current-thread))))
        (define go (make-semaphore 0))
        (define lowering
          (thread (lambda ()
                    (semaphore-wait go)
                    (parameterize ([native-memory-limit device-mark]
                                   [allocator-reading (lambda (_) #f)])
                      (collect-under-pressure! dev)))))
        (set-box! lowerer lowering)
        (with-race-hook die-in-the-flush
          (semaphore-post go)
          (thread-wait lowering))
        (check-true (unbox killed) "the lowering was killed inside its flush")
        (record-allocation! crossing (- device-mark gap) dev)
        (flush-tally!)
        (parameterize ([native-memory-limit device-mark]
                       [allocator-reading (lambda (_) #f)])
          (with-ledger
            (take-since-sample! a)
            (set-account-interval! a 0)
            (set-account-sample! a 0))
          (check-equal? (collections-of dev (lambda () (collect-under-pressure! dev))) 1
                        "the parked tally's bytes take the reading over the mark"))
        (with-ledger
          (set-account-interval! a #f)))
      (unaccount-all! (list parked crossing)))))
