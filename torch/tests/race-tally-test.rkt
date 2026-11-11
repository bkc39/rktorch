#lang racket/base

;; P2's laws (#508): per-thread tallies of the shared ledger counters.
(module+ test
  (require (only-in racket/dict dict-ref)
           (only-in racket/list shuffle)
           rackunit
           (only-in "../foreign.rkt" backward! cpu-device grad native-memory-unaccounted
                    native-memory-use sum zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/device-type.rkt" device)
           (only-in "../foreign/raw/accounts.rkt"
                    account-of checks-seen devices-over-floor ledger-entries live-bytes
                    account-trial note-mark! pressure-reading set-account-interval!
                    set-account-sample! set-account-trial! since-check-bytes
                    take-since-sample! with-ledger)
           (only-in (submod "../foreign/raw/accounts.rkt" private)
                    account-counters accounts raw-live)
           (only-in "../foreign/raw/cas.rkt" cas-add! cas-update!)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait!)
           (only-in "../foreign/raw/device-queries.rkt" allocator-reading)
           (only-in "../foreign/raw/memory.rkt"
                    adopted-bytes finalizer-diagnostics native-memory-use/fold
                    record-allocation! unaccount!)
           (only-in "../foreign/raw/pressure.rkt"
                    collect-under-pressure! lower-shadows! native-collect-margin
                    native-memory-limit refresh-shadows! reset-pressure-state!
                    since-check-of)
           (only-in "../foreign/raw/tallies.rkt"
                    counters-since-sample flush-tallies! flush-tally!)
           (only-in (submod "../foreign/raw/tallies.rkt" private)
                    own-pending tallies tally-limit tally-owner)
           (only-in "../foreign/structs.rkt" tensor-handle)
           (only-in "private/race-harness.rkt"
                    check-ledger churn-once! join join-all ledger-snapshot settle! spawn
                    stop! with-race-hook worker-thread)
           (only-in "private/spurious-cas.rkt" call-with-spurious-cas-failures)
           (only-in "private/within.rkt" bounded-test))

  (define mib (* 1024 1024))
  (define mark (* 512 mib))
  (define limit (tally-limit mark))

  (define (shared-live dev) (raw-live (account-of dev)))
  (define (fold-of dev) (dict-ref (native-memory-use/fold) dev 0))
  (define (pending dev) (own-pending (account-counters (account-of dev))))

  ;; a fake device, its tallies batched at `limit` as if its gate had run
  (define (batched-device index)
    (define dev (device 'cuda index))
    (note-mark! (account-of dev) mark)
    (flush-tallies!)
    dev)

  (define (forget-account! dev)
    (flush-tallies!)
    (void (cas-update! accounts (lambda (all) (hash-remove all dev)))))

  (define (unaccount-all! keys)
    (for-each unaccount! keys)
    (flush-tallies!))

  ;; a thread that charges `keys` and then waits, its deltas still pending
  (define (holding-charges dev keys nbytes #:pool [pool 'own])
    (define charged (make-semaphore 0))
    (define hold (make-semaphore 0))
    (define w
      (spawn (lambda ()
               (for ([k (in-list keys)])
                 (record-allocation! k nbytes dev))
               (semaphore-post charged)
               (semaphore-wait hold))
             #:pool pool))
    (semaphore-wait charged)
    (values w hold))

  (define (call-with-held-charge dev nbytes thunk)
    (flush-tally!)
    (define key (box 'held))
    (define-values (w hold) (holding-charges dev (list key) nbytes))
    (begin0
      (thunk)
      (semaphore-post hold)
      (join w)
      (unaccount-all! (list key))))

  (bounded-test "L1: churners killed at random leave the counter at the fold"
    (define dev (batched-device 40))
    (define rng (make-pseudo-random-generator))
    (define held (for/list ([_ (in-range 8)]) (box '())))
    (define (churn! mine)
      (let loop ([i 0])
        (define key (box i))
        (set-box! mine (cons key (unbox mine)))
        (record-allocation! key (* 512 (add1 (modulo i 7))) dev)
        (when (even? i)
          (unaccount! (car (unbox mine))))
        (loop (add1 i))))
    (define workers
      (for/list ([mine (in-list held)])
        (spawn (lambda () (churn! mine)) #:pool 'own)))
    (for ([w (in-list (shuffle workers))])
      (sleep (* 0.02 (random rng)))
      (stop! w))
    (flush-tallies!)
    (define live (shared-live dev))
    (check-true (>= live 0) (format "I1': shared live ~a is negative" live))
    (check-equal? live (fold-of dev) "I1': shared live against the fold")
    (unaccount-all! (apply append (map unbox held)))
    (check-equal? (shared-live dev) (fold-of dev))
    (forget-account! dev))

  (bounded-test "L2: a release flushed before its charge leaves the ledger exact"
    (define dev (batched-device 41))
    (define key (box 'early))
    (define-values (w hold) (holding-charges dev (list key) 4096))
    (flush-tally!)
    (unaccount! key)
    (flush-tally!)
    (check-equal? (shared-live dev) -4096 "the release reached the shared counter first")
    (flush-tallies!)
    (check-equal? (list (shared-live dev) (fold-of dev)) '(0 0)
                  "(shared live, fold) once the charge is flushed too")
    (semaphore-post hold)
    (join w)
    (forget-account! dev))

  (bounded-test "L3: a thread killed with its deltas pending loses none of them"
    (for ([pool (in-list '(#f own))] [index (in-naturals 42)])
      (define dev (batched-device index))
      (define keys (for/list ([i (in-range 5)]) (box i)))
      (define-values (w _hold) (holding-charges dev keys 4096 #:pool pool))
      (define registered (length (unbox tallies)))
      (stop! w)
      (check-equal? (shared-live dev) 0 "T6: no exit hook flushes a killed thread")
      (flush-tallies!)
      (check-equal? (shared-live dev) (* 5 4096)
                    (format "a killed ~a thread's charges after a flush" (or pool 'ordinary)))
      (check-true (< (length (unbox tallies)) registered) "the dead tally was pruned")
      (unaccount-all! keys)
      (check-equal? (shared-live dev) 0)
      (forget-account! dev)))

  (bounded-test "L4: every op leaves its tally inside the flush bounds"
    (define dev (batched-device 44))
    (define rng (make-pseudo-random-generator))
    (define keys
      (for/list ([i (in-range 500)])
        (define key (box i))
        (record-allocation! key (add1 (random (quotient limit 3) rng)) dev)
        (when (zero? (random 3 rng))
          (unaccount! key))
        (define now (pending dev))
        (check-true (< (abs (car now)) limit) (format "T2: pending live ~a" (car now)))
        (check-true (< (abs (cadr now)) 64) (format "T1: pending entries ~a" (cadr now)))
        key))
    (unaccount-all! keys)
    (define half (box 'half))
    (define whole (box 'whole))
    (record-allocation! half (quotient limit 2) dev)
    (flush-tally!)
    (unaccount! half)
    (record-allocation! whole limit dev)
    (check-equal? (pending dev) '(0 0 0 0)
                  "a tensor of the limit flushes though the pending sum stays under it")
    (unaccount-all! (list whole))
    (forget-account! dev))

  (define n 4096)

  (bounded-test "a device with no mark batches at the cap once its gate has looked"
    (define dev (device 'cuda 52))
    (define key (box 'unmarked))
    (flush-tally!)
    (parameterize ([native-memory-limit #f])
      (collect-under-pressure! dev))
    (record-allocation! key n dev)
    (check-equal? (car (pending dev)) n "the charge waits in the tally")
    (unaccount-all! (list key))
    (forget-account! dev))

  ;; the trial a sample shortens by the bytes it takes: `due` already in the
  ;; shared count, so the sample is due whatever else is pending
  (define (sampled-trial dev trial due)
    (define a (account-of dev))
    (parameterize ([native-memory-limit mark]
                   [allocator-reading (lambda (_) #f)])
      (with-ledger
        (take-since-sample! a)
        (set-account-interval! a 0)
        (set-account-trial! a (+ trial due)))
      (cas-add! (counters-since-sample (account-counters a)) due)
      (collect-under-pressure! dev)
      (begin0
        (account-trial a)
        (with-ledger
          (set-account-interval! a #f)
          (set-account-trial! a #f)))))

  (bounded-test "L5: every reader sees another thread's pending deltas"
    (settle!)
    (define dev (batched-device 45))
    (define a (account-of dev))
    (define (seen reader)
      (call-with-held-charge dev n reader))
    (define entries (ledger-entries))
    (check-equal? (seen (lambda () (dict-ref (native-memory-use) dev 0))) n
                  "native-memory-use")
    (check-equal? (seen (lambda () (dict-ref (finalizer-diagnostics) 'ledger-entries)))
                  (add1 entries)
                  "finalizer-diagnostics")
    (define (checked-since reader)
      (define before (since-check-of dev))
      (- (seen reader) before))
    (check-equal? (checked-since (lambda () (since-check-of dev))) n "since-check-of")
    (check-equal? (seen (lambda () (pressure-reading dev))) n "pressure-reading")
    (check-equal? (seen (lambda () (live-bytes a))) n "live-bytes")
    (check-equal? (checked-since (lambda () (dict-ref (checks-seen) a))) n "checks-seen")
    (check-equal? (seen (lambda () (sampled-trial dev (* 8 n) (quotient mark 32))))
                  (* 7 n)
                  "sample-allocator!")
    (check-not-false
     (seen (lambda ()
             (parameterize ([native-collect-margin 0])
               (member dev (devices-over-floor)))))
     "devices-over-floor")
    (define ((reading-of r) d)
      (and (equal? d dev) r))
    (check-equal? (seen (lambda ()
                          (parameterize ([allocator-reading (reading-of (* 3 n))])
                            (refresh-shadows!))
                          (dict-ref (native-memory-unaccounted) dev 0)))
                  (* 2 n)
                  "refresh-shadows!")
    (check-equal? (seen (lambda ()
                          (parameterize ([allocator-reading (reading-of (* 2 n))])
                            (lower-shadows!))
                          (dict-ref (native-memory-unaccounted) dev 0)))
                  n
                  "lower-shadows!")
    (check-equal? (seen (lambda () (reset-pressure-state!) (since-check-of dev))) 0
                  "reset-pressure-state!")
    (forget-account! dev))

  (bounded-test "L6: after a drain's #t the shared counter shows every drained release"
    (settle!)
    (define cpu (cpu-device))
    (void (zeros 1))
    (define dropped (for/list ([_ (in-range 8)]) (zeros 16)))
    (flush-tally!)
    (check-equal? (shared-live cpu) (fold-of cpu))
    (set! dropped #f)
    (check-true (collect-and-wait!) "the drain finished")
    (check-equal? (shared-live cpu) (fold-of cpu) "I7: the drained releases are flushed"))

  (bounded-test "L7: a thread's own interval crossing opens the gate"
    (define dev (batched-device 46))
    (define a (account-of dev))
    (define resets (box 0))
    (define (count-resets name subject)
      (when (and (eq? name 'checks-resetting) (equal? subject dev))
        (set-box! resets (add1 (unbox resets)))))
    (define key (box 'crossing))
    (parameterize ([native-memory-limit mark]
                   [allocator-reading (lambda (_) #f)])
      (collect-under-pressure! dev)
      (with-ledger
        (set-account-interval! a n)
        (set-account-sample! a (* 2 mark)))
      (record-allocation! key (* 2 n) dev)
      (check-equal? (since-check-bytes a) 0 "the crossing bytes are still pending")
      (with-race-hook count-resets
        (collect-under-pressure! dev)))
    (check-equal? (unbox resets) 1 "the pressure collection ran")
    (unaccount-all! (list key))
    (forget-account! dev))

  (bounded-test "an open gate flushes only when the reading could cross the mark"
    (define dev (batched-device 54))
    (define a (account-of dev))
    (define n 4096)
    (define (flushed-after sample)
      (call-with-held-charge
       dev n
       (lambda ()
         (define before (shared-live dev))
         (parameterize ([native-memory-limit mark]
                        [allocator-reading (lambda (_) #f)])
           (with-ledger
             (set-account-interval! a 0)
             (set-account-sample! a sample))
           (collect-under-pressure! dev))
         (- (shared-live dev) before))))
    (check-equal? (flushed-after 0) 0 "far under the mark, the other thread's bytes stay pending")
    (check-equal? (flushed-after (- mark n)) n "near the mark, the gate flushes them")
    (with-ledger
      (set-account-interval! a #f)
      (set-account-sample! a 0))
    (forget-account! dev))

  (bounded-test "L8: spurious CAS failures at the tallies lose nothing"
    (define dev (batched-device 47))
    (define start (ledger-entries))
    (define keys
      (for/list ([i (in-range 300)])
        (define key (box i))
        (call-with-spurious-cas-failures
         '(tally)
         (lambda ()
           (record-allocation! key 256 dev)
           (when (odd? i) (unaccount! key))
           (when (zero? (modulo i 7)) (flush-tally!))))
        key))
    (flush-tallies!)
    (check-equal? (shared-live dev) (fold-of dev) "I1' under spurious failures")
    (check-equal? (shared-live dev) (* 150 256) "I5: live bytes")
    (check-equal? (- (ledger-entries) start) 150 "I5: entries")
    (unaccount-all! keys)
    (forget-account! dev))

  (bounded-test "L9: the shared counter stays within the staleness bound"
    (define dev (batched-device 48))
    (define nbytes (quotient limit 4))
    (define threads 8)
    (define charged (for/vector #:length threads ([_ (in-range threads)]) (box 0)))
    (define stop? (box #f))
    (define (burst! mine)
      (let loop ()
        (define keys (for/list ([i (in-range 40)]) (box i)))
        (for ([k (in-list keys)])
          (record-allocation! k nbytes dev)
          (set-box! mine (+ (unbox mine) nbytes)))
        (for ([k (in-list keys)])
          (set-box! mine (- (unbox mine) nbytes))
          (unaccount! k))
        (unless (unbox stop?) (loop))))
    (define workers
      (for/list ([mine (in-vector charged)])
        (spawn (lambda () (burst! mine)) #:pool 'own)))
    (define worst
      (for/fold ([worst 0]) ([_ (in-range 2000)])
        (define shared (shared-live dev))
        (define truth (for/sum ([b (in-vector charged)]) (unbox b)))
        (define bound (+ (* (length (unbox tallies)) limit) (* 2 threads nbytes)))
        (max worst (- (abs (- shared truth)) bound))))
    (set-box! stop? #t)
    (join-all workers)
    (check-true (<= worst 0) (format "I4: the bound was exceeded by ~a bytes" worst))
    (flush-tallies!)
    (check-equal? (shared-live dev) 0)
    (forget-account! dev))

  (bounded-test "L10: adoptions write the shared counters directly"
    (settle!)
    (define dev (batched-device 49))
    (define key (box 'adopted))
    (flush-tally!)
    (record-allocation! key n dev #:adopt void)
    (check-equal? (list (shared-live dev) (car (pending dev))) (list n 0)
                  "(shared live, pending live) after an adopting charge")
    (unaccount-all! (list key))
    (forget-account! dev)
    (define cpu (cpu-device))
    (define p (zeros 256 256 #:requires-grad? #t))
    (backward! (sum p))
    (define bytes (* 256 256 4))
    (parameterize ([allocator-reading
                    (lambda (d) (and (equal? d cpu) (+ (live-bytes (account-of cpu)) bytes)))])
      (refresh-shadows!))
    (define g (grad p))
    (check-equal? (adopted-bytes (tensor-handle g)) bytes "the gradient was adopted")
    (flush-tally!)
    (define before (shared-live cpu))
    (tensor-free! g)
    (check-equal? (- before (shared-live cpu)) bytes
                  "an adopted release reached the shared counter unflushed")
    (void (sum p))
    (settle!))

  (bounded-test "L11: one-op threads leave no tally behind"
    (define dev (batched-device 50))
    (flush-tallies!)
    (define before (length (unbox tallies)))
    (for ([i (in-range 1000)])
      (join (spawn (lambda ()
                     (define key (box i))
                     (record-allocation! key 64 dev)
                     (unaccount! key))
                   #:pool (and (odd? i) 'own))))
    (flush-tallies!)
    (check-true (<= (length (unbox tallies)) before)
                (format "~a tallies registered after, ~a before"
                        (length (unbox tallies)) before))
    (check-equal? (shared-live dev) 0)
    (forget-account! dev))

  ;; `thread-dead?` suspends a parallel thread: one that asked it for every
  ;; tally on each flush ran an op path held open 2x slower or worse
  (bounded-test "a parallel thread's flush leaves dead tallies to the place's thread"
    (define dev (batched-device 53))
    (define gone (spawn (lambda () (record-allocation! (box 'gone) 64 dev))))
    (join gone)
    (join (spawn flush-tallies! #:pool 'own))
    (check-not-false (memf (lambda (t) (eq? (weak-box-value (tally-owner t)) (worker-thread gone)))
                           (unbox tallies))
                     "a parallel thread's flush pruned a dead tally")
    (flush-tallies!)
    (check-false (memf (lambda (t) (eq? (weak-box-value (tally-owner t)) (worker-thread gone)))
                       (unbox tallies))
                 "the place's thread prunes it")
    (forget-account! dev))

  (bounded-test "L12: eight churners at the default limit keep the ledger"
    (settle!)
    (define base (ledger-snapshot))
    (join-all
     (for/list ([i (in-range 8)])
       (spawn (lambda ()
                (define rng (make-pseudo-random-generator))
                (for ([_ (in-range 150)])
                  (churn-once! rng)))
              #:pool 'own)))
    (settle!)
    (check-ledger base #f)))
