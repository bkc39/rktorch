#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt" + @ sum zeros)
           (only-in "../foreign/raw/cas.rkt" cas-claim!)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait! on-main-os-thread?)
           (only-in "../foreign/raw/regions.rkt" with-cleanup)
           (only-in "../runtime/reaper.rkt"
                    drain-reapers! reap-after-collection! reap-when-dead! reaped wake-reapers!)
           (only-in (submod "../runtime/reaper.rkt" private) reaper-statistics reaper-threads)
           (only-in "private/race-harness.rkt"
                    check-ledger churn-once! hooks join-all ledger-snapshot
                    ledger-violations make-roster make-watch seeded-generator
                    settle! spawn stop-roles! await-roles! until-stopped
                    watch-hook watch-registered watch-released with-race-hook)
           (only-in "private/spurious-cas.rkt" call-with-spurious-cas-failures)
           (only-in "private/wedge.rkt" exit-status-of)
           (only-in "private/within.rkt" within))

  (define (garbage! n)
    (for ([_ (in-range n)])
      (zeros 8 8)))

  (define (await-box b #:seconds [seconds 30])
    (define deadline (+ (current-inexact-milliseconds) (* 1000 seconds)))
    (let poll ()
      (cond
        [(unbox b) (unbox b)]
        [(> (current-inexact-milliseconds) deadline) #f]
        [else (sleep 0.001) (poll)])))

  (define ((holding-one-reaper held go) name subject)
    (when (and (eq? name 'reaper-taken)
               (memq subject (reaper-threads))
               (cas-claim! held #f subject #:site 'gate))
      (let spin ()
        (unless (unbox go)
          (spin)))))

  (define (call-with-a-held-batch observe)
    (settle!)
    (define base (ledger-snapshot))
    (define held (box #f))
    (define go (box #f))
    (with-race-hook (holding-one-reaper held go)
      (with-cleanup
        (garbage! 256)
        (collect-garbage)
        (wake-reapers!)
        (check-true (thread? (await-box held)) "no reaper took a batch")
        (observe)
        #:finally (set-box! go #t)))
    (check-true (drain-reapers!))
    (settle!)
    (check-ledger base #f))

  (define (await-other-reapers!)
    (let quiet ([seen -1])
      (sleep 0.05)
      (define now (hash-ref (reaper-statistics) 'released))
      (unless (= now seen)
        (quiet now))))

  (test-case "L4: while a reaper holds a taken batch the counters equal the fold"
    (within 60 (lambda ()
                 (call-with-a-held-batch
                  (lambda ()
                    (await-other-reapers!)
                    (check-equal? (ledger-violations) '()))))))

  (test-case "L5: a drain waits for a batch taken before it began, and gives up at its deadline"
    (within 60 (lambda ()
                 (call-with-a-held-batch
                  (lambda ()
                    (define started (current-inexact-milliseconds))
                    (check-false (drain-reapers! #:deadline-ms 200))
                    (check-true (>= (- (current-inexact-milliseconds) started) 200)))))))

  (define (busy-for ms)
    (define until (+ (current-inexact-milliseconds) ms))
    (let spin ()
      (when (< (current-inexact-milliseconds) until)
        (spin))))

  (define reaper-pool (reaper-threads))

  (define slow-on-reapers
    (reaped 'slow-on-reapers
            (lambda (_v)
              (when (memq (current-thread) reaper-pool)
                (busy-for 1)))))

  (define ((holding target took go) name subject)
    (when (and (eq? name 'reaper-taken) (eq? subject (unbox target)))
      (set-box! took #t)
      (define until (+ (current-inexact-milliseconds) 20000))
      (let spin ()
        (unless (or (unbox go) (> (current-inexact-milliseconds) until))
          (spin)))))

  (define (drains-while-another-thread-holds-a-batch take!)
    (let attempt ([tries 20])
      (define target (box #f))
      (define took (box #f))
      (define go (box #f))
      (for ([_ (in-range 4096)])
        (reap-when-dead! slow-on-reapers (make-vector 1 0)))
      (collect-garbage)
      (define outcome
        (with-race-hook (holding target took go)
          (with-cleanup
            (define taker
              (thread (lambda () (set-box! target (current-thread)) (take!)) #:pool 'own))
            (let wait ()
              (unless (or (unbox took) (thread-dead? taker))
                (sleep 0.001)
                (wait)))
            (and (unbox took)
                 (let ([while-held (drain-reapers! #:deadline-ms 200)])
                   (set-box! go #t)
                   (thread-wait taker)
                   (list while-held (drain-reapers!))))
            #:finally (set-box! go #t))))
      (cond
        [outcome outcome]
        [(positive? tries) (check-true (drain-reapers!)) (attempt (sub1 tries))]
        [else (fail "the other thread never took a batch")])))

  (test-case "R5: a drain waits for a batch an op's first accounting took before it began"
    (within 60 (lambda ()
                 (check-equal? (drains-while-another-thread-holds-a-batch reap-after-collection!)
                               '(#f #t)
                               "(drained while the batch was held, drained after)"))))

  (test-case "R5: a drain waits for a batch a second drainer took before it began"
    (within 60 (lambda ()
                 (check-equal? (drains-while-another-thread-holds-a-batch
                                (lambda () (drain-reapers! #:deadline-ms 10000)))
                               '(#f #t)
                               "(drained while the batch was held, drained after)"))))

  (test-case "L13: a spurious CAS failure at 'reaper still releases each handle once"
    (within 60
            (lambda ()
              (settle!)
              (define base (ledger-snapshot))
              (define w (make-watch))
              (with-race-hook (watch-hook w)
                (call-with-spurious-cas-failures
                 '(reaper)
                 (lambda ()
                   (garbage! 256)
                   (collect-garbage)
                   (wake-reapers!)
                   (check-true (drain-reapers!)))))
              (settle!)
              (check-ledger base w))))

  (define (churners labels step-of)
    (define r (make-roster))
    (define workers
      (for/list ([label (in-list labels)])
        (spawn (until-stopped r label (step-of label)) #:pool 'own #:label label)))
    (await-roles! r labels)
    (values r workers))

  (define ((churn-step label))
    (churn-once! (seeded-generator 41 (list 'reaper label))))

  (test-case "L7: with eight churners, each collect-and-wait! drains #t within 500 ms"
    (within 120
            (lambda ()
              (settle!)
              (define base (ledger-snapshot))
              (define-values (r workers) (churners (build-list 8 values) churn-step))
              (define timings
                (for/list ([_ (in-range 20)])
                  (define started (current-inexact-milliseconds))
                  (define drained? (collect-and-wait!))
                  (cons drained? (- (current-inexact-milliseconds) started))))
              (stop-roles! r)
              (join-all workers)
              (settle!)
              (for ([t (in-list timings)])
                (check-true (car t) "a drain ran out of time")
                (check-true (< (cdr t) 500) (format "a drain took ~a ms" (cdr t))))
              (check-ledger base #f))))

  (define ((dropping-step _label))
    (define a (zeros 64 64))
    (sum (@ (+ a 1.0) a)))

  (define ((released-on-main seen) name _subject)
    (when (and (eq? name 'finalizer-releasing) (on-main-os-thread?))
      (set-box! seen #t)))

  (test-case "L8: while main waits, workers' tensors are released off main's OS thread"
    (within 120
            (lambda ()
              (settle!)
              (define w (make-watch))
              (define on-main (box #f))
              (with-race-hook (hooks (watch-hook w) (released-on-main on-main))
                (define-values (r workers) (churners (build-list 4 values) dropping-step))
                (void (sync/timeout 2 never-evt))
                (stop-roles! r)
                (join-all workers)
                (define share (/ (watch-released w) (max 1 (watch-registered w))))
                (check-false (unbox on-main) "a tensor was released on main's OS thread")
                (check-true (>= share 9/10)
                            (format "~a of ~a tensors released before settling"
                                    (watch-released w) (watch-registered w))))
              (settle!))))

  (define ((collecting-step _label))
    (garbage! 32)
    (when (zero? (random 64))
      (collect-garbage 'minor)))

  (test-case "L9: with collections from workers only, releases keep up"
    (within 120
            (lambda ()
              (settle!)
              (define w (make-watch))
              (with-race-hook (watch-hook w)
                (define-values (r workers) (churners (build-list 4 values) collecting-step))
                (void (sync/timeout 1 never-evt))
                (stop-roles! r)
                (void (sync (thread (lambda ()
                                      (join-all workers)
                                      (collect-garbage)
                                      (zeros 1))
                                    #:pool 'own)))
                (void (sync/timeout 1 never-evt))
                (define pending (- (watch-registered w) (watch-released w)))
                (check-true (<= pending (* 4 32))
                            (format "~a of ~a tensors still pending a second after the last op"
                                    pending (watch-registered w))))
              (settle!))))

  (define kill-mid-free-program
    '(begin
       (require (only-in torch zeros)
                (only-in torch/foreign/raw/cas cas-claim!)
                (only-in torch/runtime/reaper drain-reapers! wake-reapers!)
                (only-in (submod torch/runtime/reaper private) reaper-threads)
                (only-in torch/tests/private/race-harness
                         ledger-snapshot ledger-violations settle! with-race-hook)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define victim (box #f))
       (define killing (box #f))
       (define killed-mid-batch (box #f))
       (define released (box 0))
       (define (spin-ms ms)
         (define until (+ (current-inexact-milliseconds) ms))
         (let spin () (when (< (current-inexact-milliseconds) until) (spin))))
       (define (hook name subject)
         (cond
           [(and (eq? name 'reaper-taken)
                 (memq subject (reaper-threads))
                 (cas-claim! victim #f subject #:site 'gate))
            (define until (+ (current-inexact-milliseconds) 10000))
            (let spin ()
              (unless (or (unbox killing) (> (current-inexact-milliseconds) until))
                (spin)))
            (spin-ms 100)
            (set-box! killed-mid-batch (unbox killing))]
           [(and (eq? name 'reaper-parking) (eq? subject (unbox victim)))
            (let spin () (spin))]
           [(and (eq? name 'finalizer-releasing) (eq? (current-thread) (unbox victim)))
            (set-box! released (add1 (unbox released)))]
           [else (void)]))
       (call-with-process-deadline
        60
        (lambda ()
          (settle!)
          (define base (ledger-snapshot))
          (with-race-hook hook
            (for ([_ (in-range 256)]) (zeros 8 8))
            (collect-garbage)
            (wake-reapers!)
            (let wait () (unless (unbox victim) (sleep 0.001) (wait)))
            (thread-wait (thread (lambda ()
                                   (set-box! killing #t)
                                   (kill-thread (unbox victim)))
                                 #:pool 'own)))
          (define drained? (drain-reapers!))
          (settle!)
          (define violations (ledger-violations #:baseline base))
          (eprintf "killed mid-batch ~a, drained ~a, the killed reaper released ~a, violations ~s\n"
                   (unbox killed-mid-batch) drained? (unbox released) violations)
          (exit (if (and (unbox killed-mid-batch) drained? (positive? (unbox released))
                         (null? violations))
                    0
                    1))))))

  (test-case "L16: a reaper killed mid-batch still releases the whole batch"
    (check-equal? (exit-status-of kill-mid-free-program #:within 120) 0)))
