#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in ffi/unsafe/atomic in-atomic-mode?)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../foreign/raw/accounts.rkt" account-of)
           (only-in (submod "../foreign/raw/accounts.rkt" private) account-counters)
           (only-in "../foreign/raw/cas.rkt" cas-add! cas-claim!)
           (only-in "../foreign/raw/device-queries.rkt" allocator-reading)
           (only-in "../foreign/raw/memory.rkt" record-allocation! unaccount!)
           (only-in "../foreign/raw/pressure.rkt" collect-under-pressure! native-memory-limit)
           (only-in "../foreign/raw/tallies.rkt" flush-tallies! pending-of tally-live!)
           (only-in (submod "../foreign/raw/tallies.rkt" private)
                    registered-tallies tally-count tally-owner)
           (only-in "private/race-harness.rkt" join spawn with-race-hook worker-thread)
           (only-in "private/tally-devices.rkt"
                    mib shared-live tally-of-thread with-batched-device with-unmarked-device)
           (only-in "private/wedge.rkt" exit-status-of)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path accounts-module "../foreign/raw/accounts.rkt")
  (define-runtime-path regions-module "../foreign/raw/regions.rkt")
  (define-runtime-path tallies-module "../foreign/raw/tallies.rkt")
  (define-runtime-path wedge-module "private/wedge.rkt")

  (define n 4096)

  (define (one-op-threads! dev pool-of)
    (for ([i (in-range 1000)])
      (join (spawn (lambda ()
                     (define key (box i))
                     (record-allocation! key 64 dev)
                     (unaccount! key))
                   #:pool (pool-of i)))))

  (bounded-test "L11: one-op threads leave no tally behind"
    (with-batched-device (dev 50)
      (define before (tally-count))
      (one-op-threads! dev (lambda (i) (and (odd? i) 'own)))
      (flush-tallies!)
      (check-true (<= (tally-count) before)
                  (format "~a tallies registered after, ~a before" (tally-count) before))
      (check-equal? (shared-live dev) 0)))

  (bounded-test "short-lived threads that never flush every tally leave the registry bounded"
    (with-batched-device (dev 57)
      (define before (tally-count))
      (define flushes (box 0))
      (define (count-flushes name _subject)
        (when (eq? name 'tallies-flushing)
          (cas-add! flushes 1)))
      (with-race-hook count-flushes
        (one-op-threads! dev (lambda (i) (and (odd? i) 'own))))
      (collect-garbage)
      (check-equal? (unbox flushes) 0 "nothing flushed every tally")
      (check-true (<= (tally-count) (+ (* 2 before) 64))
                  (format "~a tallies registered after 1000 threads, ~a before" (tally-count) before))
      (check-equal? (tally-count) (length (registered-tallies)) "the count is the list's length")
      (flush-tallies!)
      (check-equal? (shared-live dev) 0 "the pruned tallies' counts were taken")))

  (bounded-test "a parallel thread's flush prunes the tallies of collected threads"
    (with-batched-device (dev 56)
      (define before (tally-count))
      (one-op-threads! dev (lambda (_) 'own))
      (collect-garbage)
      (join (spawn flush-tallies! #:pool 'own))
      (check-equal? (tally-count) (length (registered-tallies)) "the count is the list's length")
      (check-true (<= (tally-count) before)
                  (format "~a tallies registered after, ~a before" (tally-count) before))
      (check-equal? (shared-live dev) 0)))

  (bounded-test "a parallel thread's flush outside any region prunes a dead thread's tally"
    (with-batched-device (dev 53)
      (define gone (spawn (lambda () (record-allocation! (box 'gone) 64 dev))))
      (join gone)
      (join (spawn flush-tallies! #:pool 'own))
      (check-false (tally-of-thread (worker-thread gone))
                   "the dead thread, still reachable, kept its tally")
      (check-equal? (shared-live dev) 64 "its counts were taken before it was pruned")))

  (define probe-program
    `(begin
       (module probe racket/base
         (require (only-in (file ,(path->string accounts-module)) with-ledger)
                  (only-in (file ,(path->string regions-module)) with-uninterruptible)
                  (only-in (file ,(path->string tallies-module))
                           flush-tallies! fresh-counters note-counters-mark! pending-of
                           tally-live!)
                  (only-in (submod (file ,(path->string tallies-module)) private)
                           registered-tallies tally-owner)
                  (only-in (file ,(path->string wedge-module)) call-with-process-deadline))
         (define c (fresh-counters))
         (note-counters-mark! c #f)
         (define gone (thread (lambda () (tally-live! (pending-of c) 64 1))))
         (thread-wait gone)
         (define (finished? enter)
           (define done (box #f))
           (thread-wait (thread (lambda () (enter flush-tallies!) (set-box! done #t))
                                #:pool 'own))
           (unbox done))
         (define (held?)
           (for/or ([t (in-list (registered-tallies))])
             (eq? (weak-box-value (tally-owner t)) gone)))
         (define status
           (call-with-process-deadline
            20
            (lambda ()
              (cond
                [(not (finished? (lambda (flush) (with-ledger (flush))))) 3]
                [(not (finished? (lambda (flush) (with-uninterruptible (flush))))) 4]
                [(not (held?)) 5]
                [else 0]))))
         (exit status))
       (require 'probe)))

  (bounded-test "a parallel thread's flush inside a region leaves dead tallies alone"
    (check-equal? (exit-status-of probe-program #:within 40) 0
                  (string-append "3 or 4: the flush failed in the ledger or a region; 5: it pruned; "
                                 "142: wedged until its deadline; #f: killed at the bound")))

  (bounded-test "a tally is judged before its flush, so an owner dying between them loses nothing"
    (with-batched-device (dev 62)
      (define c (account-counters (account-of dev)))
      (define ready (make-semaphore 0))
      (define go (make-semaphore 0))
      (define added (make-semaphore 0))
      (define owner
        (spawn (lambda ()
                 (define p (pending-of c))
                 (semaphore-post ready)
                 (semaphore-wait go)
                 (tally-live! p n 1)
                 (semaphore-post added)
                 (semaphore-wait (make-semaphore 0)))
               #:pool 'own))
      (semaphore-wait ready)
      (define th (worker-thread owner))
      (define fired (box #f))
      (define (die-after-judgement name subject)
        (when (and (eq? name 'tally-judged)
                   (not (in-atomic-mode?))
                   (eq? (weak-box-value (tally-owner subject)) th)
                   (cas-claim! fired #f #t))
          (semaphore-post go)
          (semaphore-wait added)
          (kill-thread th)
          (thread-wait th)))
      (define before (shared-live dev))
      (with-race-hook die-after-judgement
        (flush-tallies!))
      (check-true (unbox fired) "the owner died between its judgement and its flush")
      (flush-tallies!)
      (check-equal? (- (shared-live dev) before) n "the counts it added after the judgement")
      (check-false (tally-of-thread th) "the next flush pruned it")))

  (bounded-test "gates that see different marks lower the limit once, not at every other gate"
    (with-unmarked-device (dev 63)
      (define rounds 50)
      (define gaters (box '()))
      (define flushes (box 0))
      (define (count-flushes name _subject)
        (when (and (eq? name 'tallies-flushing) (memq (current-thread) (unbox gaters)))
          (cas-add! flushes 1)))
      (define turns (list (make-semaphore 0) (make-semaphore 0)))
      (define ((gates mine next device-limit))
        (for ([_ (in-range rounds)])
          (semaphore-wait mine)
          (parameterize ([native-memory-limit device-limit]
                         [allocator-reading (lambda (_) #f)])
            (collect-under-pressure! dev))
          (semaphore-post next)))
      (with-race-hook count-flushes
        (define workers
          (list (spawn (gates (car turns) (cadr turns) (* 64 mib)) #:pool 'own)
                (spawn (gates (cadr turns) (car turns) #f) #:pool 'own)))
        (set-box! gaters (map worker-thread workers))
        (semaphore-post (car turns))
        (for-each join workers))
      (check-true (<= (unbox flushes) 2)
                  (format "~a flushes of every tally in ~a alternations" (unbox flushes) rounds)))))
