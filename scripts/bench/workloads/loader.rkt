#lang racket/base

(module+ main
  (require (only-in ffi/vector f32vector-ref f32vector-set! make-f32vector)
           (only-in torch
                    adaptive-avg-pool2d backward! reshape shape tensor zeros ~>)
           (only-in torch/data/loader dataloader default-collate define-dataset)
           (only-in torch/nn Linear mse-loss parameters sgd step! zero-grads!)
           (only-in torch/vision/hymenoptera hymenoptera-root)
           (only-in torch/vision/image-folder image-folder)
           (only-in torch/vision/transforms imagenet-preprocess)
           (only-in "../../../examples/test/private/env.rkt" env-number))

  (define epochs (env-number "EPOCHS" 2))
  (define workers (env-number "WORKERS" 0))
  (define prefetch (env-number "PREFETCH" 2))
  (define batch-size (env-number "BATCH" 16))
  (define decode-work (env-number "DECODE_WORK" 4))
  (define samples (env-number "SAMPLES" 2048))
  (define side 64)
  (define sample-size (* 3 side side))

  (define (xorshift x)
    (let* ([x (bitwise-and (bitwise-xor x (arithmetic-shift x 13)) #xFFFFFFFF)]
           [x (bitwise-xor x (arithmetic-shift x -17))])
      (bitwise-and (bitwise-xor x (arithmetic-shift x 5)) #xFFFFFFFF)))

  (define (decoded i)
    (define pixels (make-f32vector sample-size))
    (for/fold ([x (add1 i)]) ([_ (in-range decode-work)])
      (for/fold ([x x]) ([k (in-range sample-size)])
        (define next (xorshift x))
        (f32vector-set! pixels k (/ (exact->inexact next) #xFFFFFFFF))
        next))
    pixels)

  (define-dataset synthetic (n)
    #:length n
    #:ref (i) (decoded i))

  (define (one-tensor items)
    (define n (length items))
    (define flat (make-f32vector (* n sample-size)))
    (for ([item (in-list items)] [j (in-naturals)])
      (define pixels (car item))
      (for ([k (in-range sample-size)])
        (f32vector-set! flat (+ (* j sample-size) k) (f32vector-ref pixels k))))
    (reshape (tensor flat) n 3 side side))

  (define (loader-over ds collate)
    (if (zero? workers)
        (dataloader ds #:batch-size batch-size #:shuffle? #t #:collate collate)
        (dataloader ds #:batch-size batch-size #:shuffle? #t #:collate collate
                    #:workers workers #:prefetch prefetch)))

  (define loader
    (if (equal? (getenv "DATASET") "folder")
        (~> (hymenoptera-root)
            (build-path "train")
            (image-folder #:transform imagenet-preprocess)
            (loader-over default-collate))
        (loader-over (synthetic samples) one-tensor)))

  (define head (Linear 3 2))
  (define opt (sgd (parameters head) #:lr 0.01))

  (define (train-step! x)
    (define n (car (shape x)))
    (zero-grads! opt)
    (~> (adaptive-avg-pool2d x 1) (reshape n 3) head (mse-loss (zeros n 2)) backward!)
    (step! opt))

  (for* ([_ (in-range epochs)] [vs (in-values-sequence loader)])
    (train-step! (car vs))))
