#lang racket/base

(module+ test
  (require (only-in racket/match match)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in threading ~>)
           (only-in "../data/loader.rkt" dataloader in-dataloader tensor-dataset)
           (only-in "../main.rkt"
                    * autocast-enabled? backward! cat cpu-device cuda-available? cuda-device item
                    manual-seed! randn reshape sum tensor->list with-autocast
                    with-default-device)
           (only-in "../nn.rkt"
                    Linear grad-scaler grad-scaler-scale grad-scaler-scale-loss grad-scaler-step!
                    grad-scaler-unscale! grad-scaler-update! layer-copy mse-loss parameters sgd
                    zero-grads!)
           "../train.rkt"
           (only-in "../train/private/precision.rkt" manual-optimizer-step! window-step!)
           (only-in "../train/private/state.rkt" fresh-run set-run-grad-scaler!)
           (only-in "private/python-env.rkt"
                    call-with-python-env check-training-twin python-cuda-available?
                    python-torch-available?)
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

  (define-trainer Amp (net losses inf-at)
    (define-init (model sink [at #f]) (set! net model) (set! losses sink) (set! inf-at at))
    (define-optimizers (sgd (parameters net) #:lr 0.05 #:momentum 0.9))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (set-box! losses (cons (item loss) (unbox losses)))
      (if (eqv? inf-at (run-global-step run)) (* loss +inf.0) loss)))

  (define ((fit-on scales [inf-at #f]) _device)
    (with-default-device (cuda-device)
      (manual-seed! 0)
      (define net (Linear 8 4))
      (define losses (box '()))
      (define r
        (fit (Amp net losses inf-at)
             (dataloader (tensor-dataset (randn 32 8) (randn 32 4)) #:batch-size 8)
             #:max-epochs 2 #:logger #f #:device (cuda-device) #:precision '16-mixed
             #:gradient-clip-val 1.0))
      (set-box! scales (grad-scaler-scale (run-grad-scaler r)))
      (values (reverse (unbox losses))
              (cat (for/list ([p (in-list (parameters net))]) (reshape p -1))))))

  (define (amp-twin-ready?)
    (and (cuda-available?) (python-torch-available?) (python-cuda-available?)))

  (bounded-test "G3-L15: CUDA 16-mixed against autocast, GradScaler and clip_grad_norm_"
    (cond
      [(not (amp-twin-ready?))
       (displayln "[trainer-precision-test] G3-L15 skipped: no CUDA twin")]
      [else (check-training-twin "amp" amp-twin (fit-on (box #f)) 'cuda 5e-3)]))

  (bounded-test "G3 M3: CUDA 16-mixed skips an inf step and halves the scale, as the twin"
    (cond
      [(not (amp-twin-ready?))
       (displayln "[trainer-precision-test] G3 M3 skipped: no CUDA twin")]
      [else
       (define scales (box #f))
       (call-with-python-env
        #:env '(("RKTORCH_AMP_INF_AT" . "2"))
        (lambda () (check-training-twin "amp-inf" amp-twin (fit-on scales 2) 'cuda 5e-3)))
       (check-equal? (unbox scales) 32768.0)]))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (hand-scaled-step! s opt loss)
    (zero-grads! opt)
    (backward! (grad-scaler-scale-loss s loss))
    (grad-scaler-unscale! s opt)
    (grad-scaler-step! s opt)
    (grad-scaler-update! s))

  (define (gradless-steps step-once!)
    (manual-seed! 0)
    (define net (Linear 2 1))
    (define twin (layer-copy net))
    (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
    (define twin-opt (sgd (parameters twin) #:lr 0.1 #:momentum 0.9))
    (define r (fresh-run (Watched (box '())) 'fit (cpu-device) #:optimizers (list opt)))
    (define s (grad-scaler #:device 'cpu))
    (define twin-s (grad-scaler #:device 'cpu))
    (set-run-grad-scaler! r s)
    (define x (randn 4 2))
    (grad-scaler-scale-loss s (sum (net x)))
    (step-once! r opt)
    (step-once! r opt)
    (zero-grads! opt)
    (backward! (grad-scaler-scale-loss s (sum (net x))))
    (step-once! r opt)
    (hand-scaled-step! twin-s twin-opt (sum (twin x)))
    (check-equal? (map flat (parameters net)) (map flat (parameters twin)))
    (check-equal? (grad-scaler-scale s) (grad-scaler-scale twin-s)))

  (bounded-test "G3 H1: a window with no gradients leaves the scaler ready for the next"
    (gradless-steps (lambda (r _opt) (window-step! r #f))))

  (bounded-test "G3 H1: a manual step with no gradients leaves the scaler ready for the next"
    (gradless-steps manual-optimizer-step!))

  (define-trainer Gappy (net)
    (define-init (model) (set! net model))
    (define-optimizers (sgd (parameters net) #:lr 0.05 #:momentum 0.9))
    (define-training-step (x y)
      (match (run-batch-index run)
        [1 #f]
        [_ (mse-loss (net x) y)])))

  (define (gappy-loader)
    (dataloader (tensor-dataset (randn 32 8) (randn 32 4)) #:batch-size 8))

  (bounded-test "G3 H1: under CUDA 16-mixed, a batch returning #f skips its step and update"
    (cond
      [(not (cuda-available?)) (displayln "[trainer-precision-test] G3 H1 fit skipped: no CUDA")]
      [else
       (with-default-device (cuda-device)
         (manual-seed! 0)
         (define net (Linear 8 4))
         (define twin (layer-copy net))
         (define loader (gappy-loader))
         (define r (fit (Gappy net) loader #:max-epochs 1 #:logger #f #:device (cuda-device)
                        #:precision '16-mixed))
         (define opt (sgd (parameters twin) #:lr 0.05 #:momentum 0.9))
         (define s (grad-scaler))
         (for ([(x y) (in-dataloader loader)]
               [i (in-naturals)]
               #:unless (= i 1))
           (hand-scaled-step!
            s opt (with-autocast #:device 'cuda #:dtype 'float16 (mse-loss (twin x) y))))
         (check-equal? (run-global-step r) 4)
         (check-equal? (map flat (parameters net)) (map flat (parameters twin)))
         (check-equal? (grad-scaler-scale (run-grad-scaler r)) (grad-scaler-scale s)))])))

