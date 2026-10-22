#lang racket/base

(module+ test
  (require (only-in racket/match match)
           rackunit
           (only-in torch
                    accelerator-if-available cpu-device cuda-device device device?
                    device-type mps-available? mps-device mps-if-available tensor)
           (only-in torch/foreign/accelerator
                    make-mps-probe mps-usable? pick-accelerator)
           (only-in torch/foreign/error exn:fail:rktorch:oom)
           (only-in torch/foreign/raw/fault native-fault?))

  (define expected (getenv "RKTORCH_EXPECT_ACCELERATOR"))

  (define (skipped why)
    (printf "[accelerator-test] skipped: ~a\n" why))

  (define (logged-during thunk)
    (define isolated (make-logger))
    (define receiver (make-log-receiver isolated 'warning 'rktorch-device))
    (define result (parameterize ([current-logger isolated]) (thunk)))
    (values result
            (for/list ([event (in-producer (λ () (sync/timeout 0 receiver)) #f)])
              (cons (vector-ref event 0) (vector-ref event 1)))))

  (define (counted-probe outcome)
    (define calls 0)
    (values (λ () (set! calls (add1 calls)) (outcome calls))
            (λ () calls)))

  (define (out-of-memory _call)
    (raise (exn:fail:rktorch:oom
            "tr_tensor_to: MPS backend out of memory (MPS allocated: 0 bytes)"
            (current-continuation-marks))))

  (define (pick-with-mps probe)
    (pick-accelerator #:cuda? #f #:mps? #t #:probe probe))

  (define (finished-within seconds worker)
    (and (sync/timeout seconds (thread-dead-evt worker)) #t))

  (test-case "an MPS that cannot allocate falls back to the CPU, probed once and warned once"
    (define-values (run calls) (counted-probe out-of-memory))
    (define probe (make-mps-probe run))
    (define-values (picks logged)
      (logged-during (λ () (for/list ([_ (in-range 3)]) (pick-with-mps probe)))))
    (check-equal? picks (list (cpu-device) (cpu-device) (cpu-device)))
    (check-equal? (calls) 1)
    (check-equal? (map car logged) '(warning))
    (check-regexp-match #rx"cannot allocate.*MPS backend out of memory" (cdar logged)))

  (test-case "an MPS that allocates is picked, probed once, and nothing is logged"
    (define-values (run calls) (counted-probe void))
    (define probe (make-mps-probe run))
    (define-values (picks logged)
      (logged-during (λ () (list (pick-with-mps probe) (pick-with-mps probe)))))
    (check-equal? picks (list (mps-device) (mps-device)))
    (check-equal? (calls) 1)
    (check-equal? logged '()))

  (test-case "no probe runs when CUDA wins or MPS is not advertised"
    (define-values (run calls) (counted-probe out-of-memory))
    (define probe (make-mps-probe run))
    (check-equal? (pick-accelerator #:cuda? #t #:mps? #t #:probe probe) (cuda-device))
    (check-equal? (pick-accelerator #:cuda? #f #:mps? #f #:probe probe) (cpu-device))
    (check-equal? (calls) 0))

  (test-case "a probe that fails other than by OOM falls back to the CPU and logs an error"
    (for ([failure (list (λ () (error 'probe "a Racket-side bug"))
                         (λ () (raise 'not-an-exn)))])
      (define-values (pick logged)
        (logged-during (λ () (pick-with-mps (make-mps-probe failure)))))
      (check-equal? pick (cpu-device))
      (check-equal? (map car logged) '(error))))

  (test-case "a native fault in the probe reaches the prober, its waiters and later callers"
    (define entered (make-semaphore 0))
    (define gate (make-semaphore 0))
    (define-values (run calls)
      (counted-probe (λ (_call)
                       (semaphore-post entered)
                       (semaphore-wait gate)
                       (raise (exn:fail "invalid memory reference.  Some debugging context lost"
                                        (current-continuation-marks))))))
    (define probe (make-mps-probe run))
    (define (ask)
      (define answer (make-channel))
      (thread (λ () (channel-put answer (with-handlers ([(λ (_) #t) values])
                                          (pick-with-mps probe)))))
      answer)
    (parameterize ([current-logger (make-logger)])
      (define prober (ask))
      (check-not-false (sync/timeout 5 entered))
      (define waiter (ask))
      (check-false (sync/timeout 0.1 waiter) "a waiter answered while the probe was still running")
      (semaphore-post gate)
      (for ([answer (list prober waiter (ask))])
        (check-pred native-fault? (sync/timeout 5 answer))))
    (check-equal? (calls) 1))

  (test-case "threads asking at once share one probe"
    (define entered (make-semaphore 0))
    (define gate (make-semaphore 0))
    (define-values (run calls)
      (counted-probe (λ (n) (semaphore-post entered) (semaphore-wait gate) (out-of-memory n))))
    (define probe (make-mps-probe run))
    (define picks (make-vector 4 #f))
    (define (worker i)
      (thread (λ () (vector-set! picks i (pick-with-mps probe)))))
    (define first-worker (worker 0))
    (check-not-false (sync/timeout 5 entered))
    (define waiters (for/list ([i (in-range 1 4)]) (worker i)))
    (check-false (sync/timeout 0.1 (apply choice-evt (map thread-dead-evt waiters)))
                 "a waiter answered while the probe was still running")
    (semaphore-post gate)
    (check-true (for/and ([w (in-list (cons first-worker waiters))])
                  (finished-within 5 w)))
    (check-equal? (vector->list picks) (list (cpu-device) (cpu-device) (cpu-device) (cpu-device)))
    (check-equal? (calls) 1))

  (test-case "a break sent to the probing thread does not poison the answer"
    (define entered (make-semaphore 0))
    (define gate (make-semaphore 0))
    (define-values (run calls)
      (counted-probe (λ (_call) (semaphore-post entered) (semaphore-wait gate))))
    (define probe (make-mps-probe run))
    (define prober
      (thread (λ () (with-handlers ([exn:break? void]) (pick-with-mps probe)))))
    (check-not-false (sync/timeout 5 entered))
    (break-thread prober)
    (semaphore-post gate)
    (check-true (finished-within 5 prober))
    (define next (make-channel))
    (define asker (thread (λ () (channel-put next (pick-with-mps probe)))))
    (check-equal? (sync/timeout 5 next) (mps-device))
    (check-equal? (calls) 1)
    (kill-thread asker))

  (test-case "a waiter whose probing thread is killed probes again"
    (define entered (make-semaphore 0))
    (define-values (run calls)
      (counted-probe (λ (n)
                       (semaphore-post entered)
                       (sync (if (= n 1) never-evt always-evt)))))
    (define probe (make-mps-probe run))
    (define doomed (make-custodian))
    (parameterize ([current-custodian doomed])
      (thread (λ () (pick-with-mps probe))))
    (check-not-false (sync/timeout 5 entered))
    (define answer (make-channel))
    (define waiter (thread (λ () (channel-put answer (pick-with-mps probe)))))
    (check-false (finished-within 0.1 waiter) "the waiter answered before the probe did")
    (custodian-shutdown-all doomed)
    (check-equal? (sync/timeout 5 answer) (mps-device))
    (check-equal? (calls) 2)
    (kill-thread waiter))

  (test-case "a break landing between the claim and the probe leaves no claim behind"
    (define (probe-under-breaks count)
      (for/fold ([unrun '()]) ([_ (in-range count)])
        (for/sum ([k (in-range (random 64))]) k)
        (define ran? (box #f))
        (define probe (make-mps-probe (λ () (set-box! ran? #t))))
        (with-handlers ([exn:break? (λ (_) (if (unbox ran?) unrun (cons probe unrun)))])
          (parameterize-break #t (mps-usable? probe))
          unrun)))
    (define result (make-channel))
    (define worker
      (parameterize-break #f
        (thread (λ ()
                  (channel-put result (probe-under-breaks 30000))
                  (sync never-evt)))))
    (define deadline (+ (current-inexact-monotonic-milliseconds) 30000))
    (define unrun
      (let storm ()
        (match (sync/timeout 0 result)
          [#f #:when (< (current-inexact-monotonic-milliseconds) deadline)
              (break-thread worker)
              (sleep 0)
              (storm)]
          [answer answer])))
    (define (answered-within seconds probe)
      (define answer (make-channel))
      (define asker (thread (λ () (channel-put answer (mps-usable? probe)))))
      (begin0 (sync/timeout seconds answer) (kill-thread asker)))
    (check-not-false unrun "the probing thread did not finish within 30 s")
    (check-equal? (for/sum ([probe (in-list (or unrun '()))])
                    (if (answered-within 0.05 probe) 0 1))
                  0
                  "another thread waited on a claim its living holder abandoned")
    (kill-thread worker))

  (test-case "the real probe's OOM is caught and warned once"
    (cond
      [(not (mps-available?))
       (skipped "no Metal device is advertised; the OOM path runs on macOS 14")]
      [(not (equal? expected "cpu"))
       (skipped "RKTORCH_EXPECT_ACCELERATOR is not cpu, so MPS is not known to fail here")]
      [else
       (define-values (usable logged) (logged-during (λ () (mps-usable? (make-mps-probe)))))
       (check-false usable)
       (check-equal? (map car logged) '(warning))]))

  (test-case "accelerator-if-available answers what RKTORCH_EXPECT_ACCELERATOR names"
    (define picked (accelerator-if-available))
    (check-equal? (device (tensor '(1 2 3) #:device picked)) picked)
    (cond
      [(not expected)
       (check-pred device? picked)
       (check-pred device? (mps-if-available))
       (skipped "RKTORCH_EXPECT_ACCELERATOR is unset, so any device is right")]
      [else
       (check-equal? (device-type picked) (string->symbol expected))
       (check-equal? (device-type (mps-if-available))
                     (if (equal? expected "mps") 'mps 'cpu))])))
