#lang racket/base

;; What the no-grad trough is worth on a real sampling loop (#145), the MPS
;; counterpart of the CUDA sampler row in PR #147: a DDPM reverse process run
;; under with-no-grad, once with the trough as shipped and once with it off.
;;
;;   BASE=64 BATCH=64 STEPS=120 racket scripts/bench-sampler-trough.rkt
;;
;; The backstop is left on in BOTH arms, pinned to a fraction this machine can
;; actually hold, so the only difference between them is the no-grad trough and
;; the "off" arm still cannot run the host into swap.

(require racket/format
         torch
         torch/vision/diffusion)

(define (env name default) (or (getenv name) default))
(define BASE (string->number (env "BASE" "64")))
(define BATCH (string->number (env "BATCH" "64")))
(define STEPS (string->number (env "STEPS" "120")))
(define FRACTION (string->number (env "FRACTION" "1/3")))

(define mib (* 1024 1024))
(define never (expt 2 60))

(define (diagnostic key) (cdr (assq key (finalizer-diagnostics))))
(define (mps key) (quotient (cdr (assq key (mps-memory-info))) mib))
(define (ledger-mib)
  (for/sum ([e (in-list (native-memory-use))]) (quotient (cdr e) mib)))

(define (reverse-step net x t beta alpha alpha-bar device)
  (define n (car (shape x)))
  (define ts (full t n #:dtype 'int64 #:device device))
  (define eps (net x ts #f))
  (define mean-term (div (sub x (mul eps (/ beta (sqrt (- 1.0 alpha-bar)))))
                         (sqrt alpha)))
  (if (zero? t)
      mean-term
      (add mean-term (mul (randn-like x) (sqrt beta)))))

(define (arm label trough? device net sched)
  (define betas (tensor->list (schedule-betas sched)))
  (define alphas (tensor->list (schedule-alphas sched)))
  (define alpha-bars (tensor->list (schedule-alpha-bars sched)))
  (define total (schedule-steps sched))
  (for ([_ (in-range 3)]) (reclaim-native-memory!))
  (define minors0 (diagnostic 'trough-minors))
  (define troughs0 (diagnostic 'trough-collections))
  (define back0 (diagnostic 'pressure-collections))
  (define t0 (current-inexact-milliseconds))
  (define-values (peak-alloc peak-ledger)
    (parameterize ([native-memory-fraction FRACTION]
                   [native-memory-limit #f]
                   [native-collect-margin (if trough? #f never)])
      (with-default-device device
        (let loop ([x (randn BATCH 3 32 32)] [k 0] [pa 0] [pl 0])
          (cond
            [(>= k STEPS) (values pa pl)]
            [else
             (define t (- total 1 k))
             (define next
               (with-no-grad
                 (reverse-step net x t (list-ref betas t) (list-ref alphas t)
                               (list-ref alpha-bars t) device)))
             (loop next (add1 k)
                   (max pa (mps 'allocated))
                   (max pl (ledger-mib)))])))))
  (define secs (/ (- (current-inexact-milliseconds) t0) 1000.0))
  (printf "~a: peak allocated ~a MiB, peak ledger ~a MiB, driver ~a MiB, ~a s (~a s/step), minors ~a, full ~a, backstop ~a\n"
          label peak-alloc peak-ledger (mps 'driver-allocated)
          (~r secs #:precision '(= 1))
          (~r (/ secs STEPS) #:precision '(= 3))
          (- (diagnostic 'trough-minors) minors0)
          (- (diagnostic 'trough-collections) troughs0)
          (- (diagnostic 'pressure-collections) back0))
  (flush-output)
  peak-alloc)

(module+ main
  (unless (mps-available?)
    (error 'bench-sampler-trough "needs an MPS device"))
  (define device (mps-device))
  (manual-seed! 0)
  (printf "base=~a batch=~a steps=~a fraction=~a, mps recommended-max ~a MiB\n"
          BASE BATCH STEPS FRACTION (mps 'recommended-max))
  (define net (with-default-device device (UNet #:base BASE)))
  (define sched (linear-schedule))
  (define off (arm "trough off " #f device net sched))
  (define on (arm "trough on  " #t device net sched))
  (printf "no-grad trough cuts the sampler's peak from ~a MiB to ~a MiB (~ax)\n"
          off on (~r (/ off (max on 1)) #:precision '(= 2))))
