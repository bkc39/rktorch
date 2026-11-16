#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign/raw/accounts.rkt" account-of)
           (only-in (submod "../foreign/raw/accounts.rkt" private) account-counters)
           (only-in "../foreign/raw/cas.rkt" cas-add! cas-claim!)
           (only-in "../foreign/raw/memory.rkt" record-allocation! unaccount!)
           (only-in "../foreign/raw/phantoms.rkt" charge-epoch)
           (only-in (submod "../foreign/raw/phantoms.rkt" private) young-count-of)
           (only-in "../foreign/raw/tallies.rkt"
                    flush-tallies! flush-tally! pending-of tally-live!)
           (only-in "private/race-harness.rkt" join spawn with-race-hook worker-thread)
           (only-in "private/tally-devices.rkt"
                    kib phantoms-of spin-until tally-of-thread with-batched-device)
           (only-in "private/within.rkt" bounded-test))

  (define young (* 8 kib))

  (define (young-of dev) (young-count-of (phantoms-of dev)))

  (define (own-pending dev)
    (pending-of (account-counters (account-of dev))))

  (bounded-test "a flush's #t waits for another thread's young bytes between take and count"
    (with-batched-device (dev 73)
      (flush-tally!)
      (define p (own-pending dev))
      (tally-live! p young 1 young (charge-epoch))
      (define go (box #f))
      (define held (box 'waiting))
      (define resume (box #f))
      (define flushed (box #f))
      (define flusher
        (spawn (lambda ()
                 (spin-until (lambda () (unbox go)))
                 (flush-tallies!))
               #:pool 'own))
      (define flusher-thread (worker-thread flusher))
      (define (hold-the-young name subject)
        (when (and (eq? name 'young-taken)
                   (eq? subject p)
                   (eq? (current-thread) flusher-thread)
                   (cas-claim! held 'waiting 'parked))
          (spin-until (lambda () (unbox resume)))))
      (define early
        (with-race-hook hold-the-young
          (define watcher
            (spawn (lambda ()
                     (spin-until (lambda () (eq? (unbox held) 'parked)))
                     (spin-until (lambda () (unbox flushed)) #:seconds 1)
                     (begin0 (unbox flushed)
                       (set-box! resume #t)))
                   #:pool 'own))
          (set-box! go #t)
          (spin-until (lambda () (eq? (unbox held) 'parked)))
          (flush-tally!)
          (set-box! flushed (young-of dev))
          (begin0 (join watcher)
            (join flusher))))
      (check-eq? (unbox held) 'parked "another thread's flush held the young bytes")
      (check-false early "the owner's flush returned while the young bytes were held")
      (check-equal? (unbox flushed) young "the young bytes were counted at the flush's return")
      (tally-live! p (- young) -1 0)
      (flush-tally!)))

  (bounded-test "a flush that raises still lowers its tally's moving count"
    (with-batched-device (dev 75)
      (define target (phantoms-of dev))
      (define key (box 'raised))
      (define (raise-in-the-sync name subject)
        (when (and (eq? name 'phantoms-applied) (eq? subject target))
          (error 'sync-phantoms! "raised inside a flush")))
      (define owner
        (thread (lambda ()
                  (flush-tally!)
                  (with-handlers ([exn:fail? void])
                    (with-race-hook raise-in-the-sync
                      (record-allocation! key young dev)
                      (flush-tally!)))
                  (flush-tally!)
                  (unaccount! key)
                  (flush-tally!))))
      (define landed? (sync/timeout 10 (thread-dead-evt owner)))
      (unless landed?
        (kill-thread owner))
      (check-not-false landed? "the owner's next flush-tally! returned after the raise")))

  (bounded-test "a registration's prune counts a dead thread's pending young bytes"
    (with-batched-device (dev 74)
      (define c (account-counters (account-of dev)))
      (define gone (thread (lambda () (tally-live! (pending-of c) young 1 young (charge-epoch)))))
      (thread-wait gone)
      (define pruned (box #f))
      (define flushes (box 0))
      (define (watch-the-registry name _subject)
        (cond
          [(eq? name 'tallies-pruning) (set-box! pruned #t)]
          [(eq? name 'tallies-flushing) (cas-add! flushes 1)]
          [else (void)]))
      (with-race-hook watch-the-registry
        (for ([_ (in-range 2000)] #:break (unbox pruned))
          (thread-wait (thread (lambda () (void (pending-of c)))))))
      (check-true (unbox pruned) "a registration pruned the registry")
      (check-equal? (unbox flushes) 0 "nothing flushed every tally")
      (check-false (tally-of-thread gone) "the dead thread's tally was pruned")
      (check-equal? (young-of dev) young "its pending young bytes reached the shared count"))))
