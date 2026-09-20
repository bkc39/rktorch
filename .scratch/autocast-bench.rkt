#lang racket/base
;; the DDPM UNet's training step at batch 128 on the GPU, float32 versus
;; bfloat16 autocast: peak memory and time per step, for #152 and #145
(require (only-in racket/list drop)
         torch torch/nn
         (only-in torch/vision/diffusion UNet linear-schedule q-sample))

(define dev (cuda-device))
(set-default-device! dev)
(define batch (string->number (or (getenv "BATCH") "128")))
(define steps 12)

(define (run-arm label cast?)
  (manual-seed! 0)
  (define net (to (UNet) dev))
  (define opt (adam (parameters net) #:lr 2e-4))
  (define sched (linear-schedule))
  (cuda-empty-cache!)
  (collect-garbage)
  (define times
    (for/list ([i (in-range steps)])
      (define x0 (randn batch 3 32 32))
      (define t (to-dtype (mul (rand batch) 999.0) 'int64))
      (define noise (randn-like x0))
      (define xt (q-sample sched x0 t noise))
      (define t0 (current-inexact-milliseconds))
      (zero-grads! opt)
      (define loss
        (if cast?
            (with-autocast #:device 'cuda (mse-loss (net xt t #f) noise))
            (mse-loss (net xt t #f) noise)))
      (backward! loss)
      (step! opt)
      (item loss)
      (- (current-inexact-milliseconds) t0)))
  (define stats (cuda-memory-stats))
  (printf "~a: batch ~a, ~a ms/step (median of the last ~a), peak ~a MiB\n"
          label batch
          (round (list-ref (sort (drop times 4) <) (quotient (- steps 4) 2)))
          (- steps 4)
          (quotient (cdr (assq 'peak-allocated stats)) (* 1024 1024)))
  (flush-output))

(case (getenv "ARM")
  [("bf16") (run-arm "bfloat16 autocast" #t)]
  [("fp32") (run-arm "float32" #f)]
  [else (run-arm "float32" #f) (run-arm "bfloat16 autocast" #t)])
