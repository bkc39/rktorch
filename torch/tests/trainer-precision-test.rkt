#lang racket/base

(module+ test
  (require (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt"
                    autocast-enabled? cat cuda-available? cuda-device item manual-seed! randn
                    reshape with-default-device)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/python-env.rkt"
                    check-training-twin python-cuda-available? python-torch-available?)
           (only-in "private/trainer-kit.rkt" toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path amp-twin "python/trainer_amp_twin.py")

  (define (record! sink where)
    (set-box! sink (cons (cons where (autocast-enabled? 'cpu)) (unbox sink))))

  (define-trainer Watched (net sink)
    (define-init (s) (set! net (Linear 2 1)) (set! sink s))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (record! sink 'training-step)
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (record! sink 'validation-step)
      (mse-loss (net x) y))
    (define-hook (on-train-batch-start . batch) (record! sink 'on-train-batch-start))
    (define-hook (on-before-backward loss) (record! sink 'on-before-backward))
    (define-hook (on-before-optimizer-step optimizer) (record! sink 'on-before-optimizer-step))
    (define-hook (on-train-epoch-end) (record! sink 'on-train-epoch-end)))

  (define (fit-watched precision)
    (define sink (box '()))
    (define r (fit (Watched sink) (toy-loader 4) #:val (toy-loader 2) #:max-epochs 1
                   #:logger #f #:precision precision))
    (values r (reverse (unbox sink))))

  (bounded-test "G3-L10: bf16-mixed autocasts the steps and nothing else, then restores"
    (define-values (r events) (fit-watched 'bf16-mixed))
    (for ([e (in-list events)])
      (check-equal? (cdr e) (and (memq (car e) '(training-step validation-step)) #t)
                    (format "~a" (car e))))
    (check-false (autocast-enabled? 'cpu))
    (check-equal? (run-precision r) 'bf16-mixed)
    (check-false (run-grad-scaler r)))

  (bounded-test "G3-L11: 16-mixed on the CPU runs as bf16-mixed with Lightning's warning"
    (define err (open-output-string))
    (define-values (r _events)
      (parameterize ([current-error-port err]) (fit-watched '16-mixed)))
    (check-equal? (run-precision r) 'bf16-mixed)
    (check-false (run-grad-scaler r))
    (check-regexp-match #rx"bf16-mixed" (get-output-string err)))

  (bounded-test "G3-L11: auto-mixed on the CPU resolves to bf16-mixed"
    (define-values (r _events) (fit-watched 'auto-mixed))
    (check-equal? (run-precision r) 'bf16-mixed))

  (define-trainer FusedClipped (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1 #:fused? #t))
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "G3-L16: under CUDA 16-mixed, a fused optimizer with clipping raises"
    (cond
      [(not (cuda-available?)) (displayln "[trainer-precision-test] G3-L16 skipped: no CUDA")]
      [else
       (check-exn #rx"does not allow for gradient clipping"
                  (lambda ()
                    (fit (FusedClipped) (toy-loader 4) #:max-epochs 1 #:logger #f
                         #:device (cuda-device) #:precision '16-mixed
                         #:gradient-clip-val 1.0)))]))

  (define-trainer Amp (net losses)
    (define-init (model sink) (set! net model) (set! losses sink))
    (define-optimizers (sgd (parameters net) #:lr 0.05 #:momentum 0.9))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (set-box! losses (cons (item loss) (unbox losses)))
      loss))

  (define (fit-on _device)
    (with-default-device (cuda-device)
      (manual-seed! 0)
      (define net (Linear 8 4))
      (define losses (box '()))
      (fit (Amp net losses) (dataloader (tensor-dataset (randn 32 8) (randn 32 4)) #:batch-size 8)
           #:max-epochs 2 #:logger #f #:device (cuda-device) #:precision '16-mixed
           #:gradient-clip-val 1.0)
      (values (reverse (unbox losses))
              (cat (for/list ([p (in-list (parameters net))]) (reshape p -1))))))

  (bounded-test "G3-L15: CUDA 16-mixed against autocast, GradScaler and clip_grad_norm_"
    (cond
      [(not (and (cuda-available?) (python-torch-available?) (python-cuda-available?)))
       (displayln "[trainer-precision-test] G3-L15 skipped: no CUDA twin")]
      [else (check-training-twin "amp" amp-twin fit-on 'cuda 5e-3)])))
