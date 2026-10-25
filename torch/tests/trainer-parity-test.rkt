#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt"
                    dataloader dataloader-length in-dataloader tensor-dataset)
           "../main.rkt"
           "../nn.rkt"
           "../train.rkt"
           (only-in "../vision/mnist.rkt" load-mnist-fixture)
           "private/python-env.rkt")

  (define-layer convnet (c1 c2 f1 f2)
    #:init ()
    (set! c1 (Conv2d 1 16 3))
    (set! c2 (Conv2d 16 32 3))
    (set! f1 (Linear 800 128))
    (set! f2 (Linear 128 10))
    #:forward (x)
    (~> x
        c1 relu (max-pool2d 2)
        c2 relu (max-pool2d 2)
        (flatten 1) f1 relu
        f2))

  (define epochs 2)
  (define accumulate 2)

  (define (mnist-loader)
    (define-values (xs ys) (load-mnist-fixture))
    (dataloader (tensor-dataset xs ys) #:batch-size 40))

  (define (flat-parameters net)
    (cat (for/list ([p (in-list (parameters net))]) (reshape p -1))))

  (define-trainer Mnist (net losses)
    (define-init (model sink) (set! net model) (set! losses sink))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.05 #:momentum 0.9))
      (values opt (cosine-annealing-lr opt #:t-max epochs)))
    (define-training-step (x y)
      (define loss (cross-entropy (net x) y))
      (set-box! losses (cons (item loss) (unbox losses)))
      loss))

  (define (fit-on device)
    (with-default-device device
      (manual-seed! 0)
      (define net (convnet))
      (define losses (box '()))
      (fit (Mnist net losses) (mnist-loader) #:max-epochs epochs
           #:accumulate-grad-batches accumulate #:device device #:logger #f)
      (values (reverse (unbox losses)) (flat-parameters net))))

  (define (hand-loop device)
    (with-default-device device
      (manual-seed! 0)
      (define net (convnet))
      (define opt (sgd (parameters net) #:lr 0.05 #:momentum 0.9))
      (define schedule (cosine-annealing-lr opt #:t-max epochs))
      (define loader (mnist-loader))
      (define last (sub1 (dataloader-length loader)))
      (define losses
        (for*/list ([_ (in-range epochs)]
                    [(x y i) (in-parallel (in-dataloader loader) (in-naturals))])
          (define loss (cross-entropy (net x) y))
          (when (zero? (modulo i accumulate)) (zero-grads! opt))
          (backward! (/ loss accumulate))
          (when (or (~> i add1 (modulo accumulate) zero?) (= i last)) (step! opt))
          (when (= i last) (step! schedule))
          (item loss)))
      (values losses (flat-parameters net))))

  (test-case "L-P1: fit leaves the parameters a hand-written loop leaves, bit for bit"
    (define-values (fit-losses fit-params) (fit-on 'cpu))
    (define-values (hand-losses hand-params) (hand-loop 'cpu))
    (check-equal? fit-losses hand-losses)
    (check-equal? (tensor->list fit-params) (tensor->list hand-params)))

  (test-case "L-P2: fit against the PyTorch twin"
    (cond
      [(not (python-torch-available?))
       (displayln "[trainer-parity-test] L-P2 skipped: python3 `torch` not available")]
      [else
       (check-training-twin "trainer" "../torch/tests/python/trainer_twin.py" fit-on 'cpu tol)
       (when (and (cuda-available?) (python-cuda-available?))
         (check-training-twin "trainer" "../torch/tests/python/trainer_twin.py" fit-on
                              'cuda 5e-3))])))
