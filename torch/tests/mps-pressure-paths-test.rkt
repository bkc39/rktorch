#lang racket/base

;; The #145 collection paths that a training loop never reaches, driven on a
;; real MPS device rather than against stubbed device queries: the no-grad
;; trough (a sampling loop), the backstop fed by the MPS capacity query, the
;; hysteresis back-off, `reaccount!` on an in-place move, and the single-
;; collector claim. Self-skips where there is no MPS, as the device suite does.
;;
;; Every case asserts that the counter it targets *moved*: a path that silently
;; stopped firing would pass a test that only checked for the absence of a
;; crash.

(module+ test
  (require rackunit
           (only-in "../foreign.rkt"
                    add cpu-device div finalizer-diagnostics full item
                    mean mps-available? mps-device mps-memory-info mul
                    native-memory-limit native-memory-use randn randn-like
                    reclaim-native-memory! shape sub tensor->list tensor-device
                    with-default-device with-no-grad zeros)
           (only-in (submod "../foreign.rkt" unsafe) to!)
           (only-in "../foreign/raw/memory.rkt" native-memory-use/fold)
           (only-in "../foreign/raw/pressure.rkt"
                    native-collect-budget native-collect-margin
                    native-memory-fraction reset-pressure-state!)
           (only-in "../nn.rkt" Linear Sequential parameters)
           (only-in "../vision/diffusion.rkt"
                    UNet linear-schedule schedule-alpha-bars schedule-alphas
                    schedule-betas schedule-steps))

  (define mib (* 1024 1024))
  (define never (expt 2 60))

  (define (diagnostic key) (cdr (assq key (finalizer-diagnostics))))
  (define (minors) (diagnostic 'trough-minors))
  (define (troughs) (diagnostic 'trough-collections))
  (define (backstops) (diagnostic 'pressure-collections))

  (define (device-bytes dev)
    (cond [(assoc dev (native-memory-use)) => cdr] [else 0]))

  ;; the ledger and the policy both start from a known state
  (define (settle!)
    (for ([_ (in-range 3)]) (reclaim-native-memory!))
    (reset-pressure-state!))

  ;; --- the model and the reverse process the no-grad cases drive ---

  ;; small enough to stay quick, wide enough that one forward's intermediates
  ;; clear the 256 MiB floor of the shipped margin at the batch below
  (define (small-unet)
    (UNet #:base 64 #:mults '(1 2) #:blocks 1 #:attention '(16) #:dropout 0))

  (define sample-batch 48)

  ;; One DDPM reverse step. Every image in the batch shares a timestep, so the
  ;; schedule's per-t coefficients are plain Racket reals pulled once, which
  ;; keeps this loop to the ops the trough path actually cares about.
  (define (reverse-step net x t beta alpha alpha-bar device)
    (define n (car (shape x)))
    (define ts (full t n #:dtype 'int64 #:device device))
    (define eps (net x ts #f))
    (define coeff (/ beta (sqrt (- 1.0 alpha-bar))))
    (define mean-term (div (sub x (mul eps coeff)) (sqrt alpha)))
    (if (zero? t)
        mean-term
        (add mean-term (mul (randn-like x) (sqrt beta)))))

  (define (run-sampler net sched steps device)
    (define betas (tensor->list (schedule-betas sched)))
    (define alphas (tensor->list (schedule-alphas sched)))
    (define alpha-bars (tensor->list (schedule-alpha-bars sched)))
    (define total (schedule-steps sched))
    (for/fold ([x (randn sample-batch 3 32 32 #:device device)])
              ([k (in-range steps)])
      (define t (- total 1 k))
      (with-no-grad
        (reverse-step net x t
                      (list-ref betas t) (list-ref alphas t)
                      (list-ref alpha-bars t) device))))

  ;; ------------------------------------------------------------------
  (cond
    [(not (mps-available?))
     (displayln "[mps-pressure-paths] no MPS device; skipping")]
    [else
     (define device (mps-device))

     (test-case "the MPS capacity query answers, and it is what the mark uses"
       (define info (mps-memory-info))
       (define capacity (cdr (assq 'recommended-max info)))
       (check-true (positive? capacity)
                   "recommendedMaxWorkingSetSize must be known on MPS")
       ;; No limit, no margin: the only trigger that can fire is the backstop,
       ;; and the only number it can be using is the queried capacity.
       (settle!)
       (define before (backstops))
       (define held
         (parameterize ([native-memory-limit #f]
                        [native-memory-fraction 1/200]
                        [native-collect-margin never])
           (with-default-device device
             (for/list ([_ (in-range 24)]) (randn 1024 1024)))))
       (check-equal? (length held) 24)
       (check-true (> (backstops) before)
                   "the backstop never fired off the queried MPS capacity")
       (settle!))

     (test-case "a no-grad sampler loop troughs, with a minor before the full"
       (settle!)
       (define net (with-default-device device (small-unet)))
       (define sched (linear-schedule))
       (define before-minors (minors))
       (define before-troughs (troughs))
       (define out
         (parameterize ([native-collect-margin (* 16 mib)]
                        [native-collect-budget 1])
           (run-sampler net sched 6 device)))
       (check-equal? (shape out) (list sample-batch 3 32 32))
       (check-true (> (minors) before-minors)
                   "the no-grad trough never ran a minor collection")
       (check-true (> (troughs) before-troughs)
                   "the no-grad trough never ran a full collection")
       (settle!))

     (test-case "one outermost call per step, however deep the UNet nests"
       (settle!)
       (define net (with-default-device device (small-unet)))
       (define sched (linear-schedule))
       (define steps 4)
       (define before (minors))
       (void
        (parameterize ([native-collect-margin (* 16 mib)]
                       [native-collect-budget 1])
          (run-sampler net sched steps device)))
       ;; The UNet calls dozens of sub-layers per forward. Counting those would
       ;; put the tally far above the step count; the mark must hold it to one.
       (check-equal? (- (minors) before) steps
                     "a nested layer call was counted as its own trough")
       (settle!))

     (test-case "with gradients on, the same shape is the peak, not a trough"
       (settle!)
       (define net (with-default-device device (small-unet)))
       (define before (minors))
       (void
        (parameterize ([native-collect-margin (* 16 mib)]
                       [native-collect-budget 1])
          (with-default-device device
            (define x (randn 8 3 32 32))
            (define ts (full 500 8 #:dtype 'int64))
            (for ([_ (in-range 3)]) (net x ts #f)))))
       (check-equal? (minors) before
                     "a forward with gradients on took a no-grad trough")
       (settle!))

     (test-case "an exception inside a no-grad forward leaves no mark behind"
       (settle!)
       (define net (with-default-device device (small-unet)))
       (parameterize ([native-collect-margin (* 16 mib)]
                      [native-collect-budget 1])
         (with-default-device device
           ;; a [N 3 32 32] batch is the contract; 16x16 raises inside forward
           (check-exn exn:fail?
                      (lambda ()
                        (with-no-grad
                          (net (randn 4 3 16 16)
                               (full 10 4 #:dtype 'int64) #f))))
           ;; the continuation mark unwound with the exception, so the next
           ;; outermost call is still recognised as one
           (define before (minors))
           (void (with-no-grad
                   (net (randn sample-batch 3 32 32)
                        (full 10 sample-batch #:dtype 'int64) #f)))
           (check-equal? (- (minors) before) 1
                         "the trough was lost after an exception")))
       (settle!))

     (test-case "the backstop bounds a loop that never reaches backward!"
       (settle!)
       (define mark (* 64 mib))
       ;; one "step": 32 temporaries of 4 MiB alive together, then dropped as a
       ;; batch -- a sampling loop's shape, with no backward! to trough at. Live
       ;; bytes cross the mark inside every step, so only the backstop can act.
       (define per-step (* 32 4 mib))
       (define before (backstops))
       (define high-water
         (parameterize ([native-memory-limit mark]
                        [native-collect-margin never])
           (with-default-device device
             (for/fold ([peak 0]) ([_ (in-range 12)])
               (define batch (for/list ([_ (in-range 32)]) (randn 1024 1024)))
               (define seen (max peak (device-bytes device)))
               (void (length batch))
               seen))))
       (check-true (> (backstops) before) "the backstop never fired")
       ;; 12 steps allocate 1.5 GiB in total; without the backstop the residue
       ;; accumulates. Two steps' worth is the headroom one live step needs.
       (check-true (< high-water (* 2 per-step))
                   (format "the ledger ran to ~a MiB; one step holds ~a MiB"
                           (quotient high-water mib) (quotient per-step mib)))
       (settle!))

     (test-case "a working set above the mark backs off instead of thrashing"
       (settle!)
       (define before (backstops))
       ;; 64 MiB of tensors held live against a 32 MiB mark: the mark can never
       ;; be satisfied, so the interval must double rather than collect per op
       (define kept
         (parameterize ([native-memory-limit (* 32 mib)]
                        [native-collect-margin never])
           (with-default-device device
             (define resident (for/list ([_ (in-range 16)]) (randn 1024 1024)))
             (for ([_ (in-range 40)]) (void (randn 512 512)))
             resident)))
       (check-equal? (length kept) 16)
       (define fired (- (backstops) before))
       (check-true (> fired 0) "the backstop never fired")
       (check-true (<= fired 8)
                   (format "the backstop thrashed: ~a collections" fired))
       (settle!))

     (test-case "an in-place move re-accounts, and the counters match the fold"
       (settle!)
       (define t (with-default-device device (randn 512 512)))
       (check-equal? (tensor-device t) device)
       (define on-mps (device-bytes device))
       (check-true (positive? on-mps) "the tensor was never charged to MPS")
       (void (to! t (cpu-device)))
       (check-equal? (tensor-device t) (cpu-device))
       (check-true (< (device-bytes device) on-mps)
                   "the MPS charge survived the move off the device")
       (check-true (positive? (device-bytes (cpu-device)))
                   "the CPU was never charged after the move")
       ;; the running counters are the thing this PR added; the fold is the
       ;; independent witness they are kept honest against
       (define counters
         (filter (lambda (e) (positive? (cdr e))) (native-memory-use)))
       (define folded
         (filter (lambda (e) (positive? (cdr e))) (native-memory-use/fold)))
       (check-equal? counters folded
                     "the per-device counters drifted from the entry fold")
       (settle!))

     (test-case "two threads at a trough: the work is not done twice over"
       (settle!)
       (define net (with-default-device device (small-unet)))
       (define sched (linear-schedule))
       (define before (minors))
       (define threads
         (for/list ([_ (in-range 2)])
           (thread
            (lambda ()
              (parameterize ([native-collect-margin (* 16 mib)]
                             [native-collect-budget 1])
                (void (run-sampler net sched 3 device)))))))
       (for-each thread-wait threads)
       (define fired (- (minors) before))
       ;; six outermost calls between the two threads; a claim that let both
       ;; collect at once would show up as a count above the call total
       (check-true (> fired 0) "no trough fired under concurrency")
       (check-true (<= fired 6)
                   (format "more troughs (~a) than outermost calls (6)" fired))
       (settle!))]))
