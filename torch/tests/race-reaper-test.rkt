#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt" + @ sum zeros)
           (only-in "../foreign/raw/cas.rkt" cas-claim!)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait! on-main-os-thread?)
           (only-in "../runtime/reaper.rkt" drain-reapers! wake-reapers!)
           (only-in (submod "../runtime/reaper.rkt" private) reaper-threads)
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
      (dynamic-wind
       void
       (lambda ()
         (garbage! 256)
         (collect-garbage)
         (wake-reapers!)
         (check-true (thread? (await-box held)) "no reaper took a batch")
         (observe))
       (lambda () (set-box! go #t))))
    (check-true (drain-reapers!))
    (settle!)
    (check-ledger base #f))

  (test-case "L4: while a reaper holds a taken batch the counters equal the fold"
    (within 60 (lambda ()
                 (call-with-a-held-batch
                  (lambda () (check-equal? (ledger-violations) '()))))))

  (test-case "L5: a drain waits for a batch taken before it began, and gives up at its deadline"
    (within 60 (lambda ()
                 (call-with-a-held-batch
                  (lambda ()
                    (define started (current-inexact-milliseconds))
                    (check-false (drain-reapers! #:deadline-ms 200))
                    (check-true (>= (- (current-inexact-milliseconds) started) 200)))))))

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

  (define ((collecting-step label))
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
       (define go (box #f))
       (define released (box 0))
       (define (hook name subject)
         (cond
           [(and (eq? name 'reaper-taken)
                 (memq subject (reaper-threads))
                 (cas-claim! victim #f subject #:site 'gate))
            (let spin () (unless (unbox go) (spin)))]
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
            (define killer (thread (lambda () (kill-thread (unbox victim))) #:pool 'own))
            (sleep 0.1)
            (set-box! go #t)
            (thread-wait killer))
          (define drained? (drain-reapers!))
          (settle!)
          (define violations (ledger-violations #:baseline base))
          (eprintf "drained ~a, the killed reaper released ~a, violations ~s\n"
                   drained? (unbox released) violations)
          (exit (if (and drained? (positive? (unbox released)) (null? violations)) 0 1))))))

  (test-case "L16: a reaper killed mid-batch still releases the whole batch"
    (check-equal? (exit-status-of kill-mid-free-program #:within 120) 0)))
