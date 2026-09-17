#lang racket/base

(module+ test
  (require (only-in racket/list last [take list-take])
           rackunit
           "../main.rkt"
           "../nn.rkt"
           "../vision/diffusion.rkt")

  (test-case "linear schedule: betas from start to end, alpha-bars falling to near zero"
    (define s (linear-schedule 1000))
    (check-true (schedule? s))
    (check-equal? (schedule-steps s) 1000)
    (define betas (tensor->list (schedule-betas s)))
    (check-= (car betas) 1e-4 1e-9)
    (check-= (last betas) 0.02 1e-7)
    (define abars (tensor->list (schedule-alpha-bars s)))
    (check-= (car abars) (- 1.0 1e-4) 1e-7)
    (for ([a (in-list abars)] [b (in-list (cdr abars))])
      (check-true (< b a) "alpha-bar must fall"))
    (check-true (< (last abars) 1e-4))
    (check-equal? (tensor->list (schedule-alphas (linear-schedule 3 #:beta-start 0.1 #:beta-end 0.3)))
                  '(0.8999999761581421 0.800000011920929 0.699999988079071)))

  (test-case "cosine schedule: betas capped at 0.999, alpha-bars falling from near one"
    (define s (cosine-schedule 100))
    (define betas (tensor->list (schedule-betas s)))
    ;; 0.999 rounds up in float32
    (check-true (andmap (lambda (b) (and (< 0.0 b) (<= b 0.9990001))) betas))
    (define abars (tensor->list (schedule-alpha-bars s)))
    (check-true (> (car abars) 0.99))
    (for ([a (in-list abars)] [b (in-list (cdr abars))])
      (check-true (< b a) "alpha-bar must fall"))
    (check-true (< (last abars) 0.01))
    (check-exn #rx"variance" (lambda () (linear-schedule 10 #:beta-end 1.5)))
    (check-exn #rx"^cosine-schedule: contract violation"
               (lambda () (cosine-schedule 10 #:offset -1)))
    (check-exn #rx"finite-nonnegative-real"
               (lambda () (cosine-schedule 10 #:offset +inf.0)))
    (check-exn #rx"finite-nonnegative-real"
               (lambda () (cosine-schedule 10 #:offset (expt 10 400)))))

  (test-case "q-sample mixes signal and noise by the schedule at each timestep"
    (define s (linear-schedule 10))
    (define x0 (ones 2 1 1 1))
    (define noise (full 3.0 2 1 1 1))
    (define t (tensor '(0 9) #:dtype 'int64))
    (define abars (tensor->list (schedule-alpha-bars s)))
    (define got (tensor->list (q-sample s x0 t noise)))
    (define (expected a) (+ (sqrt a) (* 3.0 (sqrt (- 1.0 a)))))
    (check-= (car got) (expected (car abars)) 1e-6)
    (check-= (cadr got) (expected (last abars)) 1e-6)
    (check-exn #rx"int64-vector" (lambda () (q-sample s x0 (arange 2) noise)))
    (check-exn #rx"one timestep per"
               (lambda () (q-sample s x0 (tensor '(3) #:dtype 'int64) noise)))
    (check-exn #rx"noise of the images' shape"
               (lambda () (q-sample s x0 t (full 3.0 1 1 1 1)))))

  (test-case "sinusoidal embedding: sines then cosines, t = 0 gives zeros then ones"
    (define e (sinusoidal-embedding (tensor '(0 1) #:dtype 'int64) 8))
    (check-equal? (tensor-shape e) '(2 8))
    (for ([dev (in-list (list (and (cuda-available?) (cuda-device))
                              (and (mps-available?) (mps-device))))]
          #:when dev)
      (check-equal? (tensor-device (sinusoidal-embedding (to (tensor '(3) #:dtype 'int64) dev) 8))
                    dev
                    "the frequency table follows the timesteps, not the default device"))
    (check-equal? (list-take (tensor->list e) 8) '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0))
    (check-= (list-ref (tensor->list e) 8) (sin 1.0) 1e-6)
    (check-exn #rx"even-positive-integer"
               (lambda () (sinusoidal-embedding (tensor '(0) #:dtype 'int64) 7)))
    (check-exn #rx"int64-vector" (lambda () (sinusoidal-embedding (arange 2) 8)))
    (check-exn #rx"int64-vector"
               (lambda () (sinusoidal-embedding (tensor '((0) (1))) 8))))

  (test-case "layers: shapes, predicates, contracts"
    (manual-seed! 0)
    (define te (TimeEmbedding 8))
    (check-true (time-embedding? te))
    (check-equal? (tensor-shape (te (tensor '(0 5) #:dtype 'int64))) '(2 32))
    (check-equal? (tensor-dtype ((to (TimeEmbedding 8) 'float64) (tensor '(0 5) #:dtype 'int64)))
                  'float64
                  "the features follow the layer's dtype")
    (define rb (ResBlock 32 64 128))
    (check-true (res-block? rb))
    (check-equal? (map car (named-parameters rb))
                  '("norm1.weight" "norm1.bias" "conv1.weight" "conv1.bias"
                    "emb.weight" "emb.bias" "norm2.weight" "norm2.bias"
                    "conv2.weight" "conv2.bias" "skip.weight" "skip.bias"))
    (check-equal? (tensor-shape (rb (randn 2 32 4 4) (randn 2 128))) '(2 64 4 4))
    (check-equal? (length (parameters (ResBlock 32 32 128))) 10
                  "no skip conv when the widths agree")
    (define ab (AttentionBlock 32))
    (check-true (attention-block? ab))
    (check-equal? (map car (named-parameters ab))
                  '("norm.weight" "norm.bias" "q.weight" "q.bias" "k.weight" "k.bias"
                    "v.weight" "v.bias" "proj.weight" "proj.bias"))
    (check-equal? (tensor-shape (ab (randn 2 32 4 4))) '(2 32 4 4))
    (check-equal? (map tensor-shape (parameters ab))
                  '((32) (32) (32 32) (32) (32 32) (32) (32 32) (32) (32 32) (32))
                  "queries, keys, values and the projection are linear maps of a token")
    (check-equal? (tensor-shape ((Downsample 32) (randn 2 32 8 8) #f)) '(2 32 4 4))
    (check-equal? (tensor-shape ((Upsample 32) (randn 2 32 4 4))) '(2 32 8 8))
    (define net (UNet #:base 32 #:mults '(1 2) #:blocks 1 #:attention '(16) #:dropout 0))
    (check-true (unet? net))
    (check-equal? (tensor-shape (net (randn 2 3 32 32) (tensor '(0 999) #:dtype 'int64) #f))
                  '(2 3 32 32))
    (define labelled (UNet #:base 32 #:mults '(1 2) #:blocks 1 #:attention '() #:classes 10))
    (check-equal? (tensor-shape (labelled (randn 2 3 32 32)
                                          (tensor '(0 999) #:dtype 'int64)
                                          (tensor '(3 10) #:dtype 'int64)))
                  '(2 3 32 32))
    (check-exn #rx"one int64 timestep and"
               (lambda () (net (randn 2 3 32 32) (tensor '(5) #:dtype 'int64) #f)))
    (check-exn #rx"one int64 label per image"
               (lambda () (labelled (randn 2 3 32 32) (tensor '(0 999) #:dtype 'int64)
                                    (tensor '(3) #:dtype 'int64))))
    (check-exn #rx"no labels otherwise"
               (lambda () (net (randn 2 3 32 32) (tensor '(0 999) #:dtype 'int64)
                               (tensor '(3 4) #:dtype 'int64))))
    (check-exn #rx"int64 timestep"
               (lambda () (net (randn 2 3 32 32) (tensor '(0.0 999.0)) #f)))
    (check-exn #rx"\\[N 3 32 32\\]"
               (lambda () (net (randn 2 3 64 64) (tensor '(0 999) #:dtype 'int64) #f)))
    (check-equal? (unet-classes labelled) 10)
    (check-false (unet-classes net))
    (check-exn #rx"multiple-of-32" (lambda () (UNet #:base 12)))
    (check-exn #rx"at-most-five-levels" (lambda () (UNet #:base 32 #:mults '(1 1 1 1 1 1))))
    (check-exn #rx"one of the levels' resolutions"
               (lambda () (UNet #:base 32 #:mults '(1 2) #:attention '(8))))
    (check-equal? (tensor-shape ((UNet #:base 32 #:mults '(2 2) #:blocks 1 #:attention '())
                                 (randn 1 3 32 32) (tensor '(5) #:dtype 'int64) #f))
                  '(1 3 32 32)
                  "a first multiplier above one widens the output layers too")
    (check-exn #rx"^ResBlock: contract violation" (lambda () (ResBlock 32 40 128)))
    (check-exn #rx"^TimeEmbedding: contract violation" (lambda () (TimeEmbedding 7))))

  (test-case "the default UNet is the DDPM paper's CIFAR-10 model"
    (define net (UNet))
    (check-equal? (for/sum ([p (in-list (parameters net))]) (numel p)) 35746307)))
