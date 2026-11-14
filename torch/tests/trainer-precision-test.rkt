#lang racket/base

(module+ test
  (require (only-in racket/match match match-define struct*)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../data/loader.rkt" dataloader in-dataloader tensor-dataset)
           (only-in "../main.rkt"
                    * autocast-cache-enabled? autocast-enabled? backward! cat cpu-device
                    cuda-available? cuda-device grad item manual-seed! randn reshape sum
                    with-autocast with-default-device)
           (only-in "../nn.rkt"
                    Linear grad-scaler grad-scaler-scale grad-scaler-scale-loss grad-scaler-step!
                    grad-scaler-unscale! grad-scaler-update! layer-copy mse-loss parameters sgd
                    zero-grads!)
           "../train.rkt"
           (only-in "../train/private/precision.rkt"
                    clipping manual-optimizer-step! window-step!)
           (only-in "../train/private/state.rkt" fresh-run set-run-grad-scaler!)
           (only-in "private/python-env.rkt"
                    call-with-python-env check-training-twin python-cuda-available?
                    python-torch-available?)
           (only-in "private/flat.rkt" flat)
           (only-in "private/trainer-kit.rkt" toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path amp-twin "python/trainer_amp_twin.py")

  (define (record! sink where)
    (set-box! sink (cons (list where (autocast-enabled? 'cpu) (autocast-cache-enabled?))
                         (unbox sink))))

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
    (check-not-false (assq 'validation-step events))
    (for ([e (in-list events)])
      (match-define (list where on? cache?) e)
      (define step? (and (memq where '(training-step validation-step)) #t))
      (check-equal? on? step? (format "~a autocasts" where))
      (check-equal? cache? (not step?) (format "G3 round 3 L1: ~a caches casts" where)))
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
    (gradless-steps (lambda (r _opt) (window-step! r #f #:backward? #t))))

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
         (check-equal? (grad-scaler-scale (run-grad-scaler r)) (grad-scaler-scale s)))]))

  (define (gradients net)
    (for/list ([p (in-list (parameters net))]) (flat (grad p))))

  (define-trainer GradWatch (net seen)
    (define-init (model sink) (set! net model) (set! seen sink))
    (define-optimizers (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-before-optimizer-step optimizer)
      (set-box! seen (cons (gradients net) (unbox seen)))))

  (struct windowed (run net opt scaler x before unscaled seen))

  (define (scaled-window backward? #:clip [clip #f])
    (manual-seed! 0)
    (define net (Linear 2 2))
    (define twin (layer-copy net))
    (define x (randn 4 2))
    (define seen (box '()))
    (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
    (define r (fresh-run (GradWatch net seen) 'fit (cpu-device) #:optimizers (list opt)))
    (define s (grad-scaler #:device 'cpu))
    (set-run-grad-scaler! r s)
    (backward! (grad-scaler-scale-loss s (sum (net x))))
    (define before (map flat (parameters net)))
    (window-step! r clip #:backward? backward?)
    (backward! (sum (twin x)))
    (windowed r net opt s x before (gradients twin) (reverse (unbox seen))))

  (bounded-test "G3 M2: on-before-optimizer-step sees unscaled gradients under a scaler"
    (match-define (struct* windowed ([unscaled unscaled] [seen seen])) (scaled-window #t))
    (check-equal? seen (list unscaled)))

  (bounded-test "G3 M1: a scaler window ending in #f runs the hook, clips, never steps"
    (match-define (windowed r net opt s x before unscaled seen)
      (scaled-window #f #:clip (clipping 1.0 'value)))
    (check-equal? (map flat (parameters net)) before "no step")
    (check-equal? (length seen) 1 "the hook still runs")
    (check-equal? (car seen)
                  (for/list ([g (in-list unscaled)]) (map (lambda (v) (* 65536.0 v)) g))
                  "the gradients stay scaled")
    (for ([v (in-list (apply append (gradients net)))])
      (check-true (<= (abs v) 1.0) "the scaled gradients are clipped, as Lightning's"))
    (check-equal? (grad-scaler-scale s) 65536.0)
    (zero-grads! opt)
    (backward! (grad-scaler-scale-loss s (sum (net x))))
    (window-step! r #f #:backward? #t)
    (manual-seed! 0)
    (define twin (Linear 2 2))
    (define twin-s (grad-scaler #:device 'cpu))
    (hand-scaled-step! twin-s (sgd (parameters twin) #:lr 0.1 #:momentum 0.9) (sum (twin x)))
    (check-equal? (map flat (parameters net)) (map flat (parameters twin))
                  "the next window steps as the first"))

  (define-trainer GapAmp (net losses)
    (define-init (model sink) (set! net model) (set! losses sink))
    (define-optimizers (sgd (parameters net) #:lr 0.05 #:momentum 0.9))
    (define-training-step (x y)
      (match (run-batch-index run)
        [1 #f]
        [_
         (define loss (mse-loss (net x) y))
         (set-box! losses (cons (item loss) (unbox losses)))
         loss])))

  (define ((fit-gap steps) _device)
    (with-default-device (cuda-device)
      (manual-seed! 0)
      (define net (Linear 8 4))
      (define losses (box '()))
      (define r
        (fit (GapAmp net losses)
             (dataloader (tensor-dataset (randn 32 8) (randn 32 4)) #:batch-size 8)
             #:max-epochs 2 #:logger #f #:device (cuda-device) #:precision '16-mixed
             #:accumulate-grad-batches 2 #:gradient-clip-val 1.0))
      (set-box! steps (run-global-step r))
      (values (reverse (unbox losses))
              (cat (for/list ([p (in-list (parameters net))]) (reshape p -1))))))

  (bounded-test "G3 M1: CUDA 16-mixed drops a window whose last batch returns #f, as Lightning"
    (cond
      [(not (amp-twin-ready?))
       (displayln "[trainer-precision-test] G3 M1 twin skipped: no CUDA twin")]
      [else
       (define steps (box #f))
       (call-with-python-env
        #:env '(("RKTORCH_AMP_GAP" . "1"))
        (lambda () (check-training-twin "amp-gap" amp-twin (fit-gap steps) 'cuda 5e-3)))
       (check-equal? (unbox steps) 4 "Lightning counts the dropped windows' steps")])))
