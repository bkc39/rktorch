#lang racket/base

(require (only-in torch
                  backward! item make-generator manual-seed! numel select to unsqueeze
                  with-default-device ~>)
         (only-in torch/data/loader dataloader define-dataset in-dataloader)
         (only-in torch/nn adam cross-entropy parameters step! zero-grads!)
         (only-in torch/scribblings/examples/05-mnist accuracy convnet pick-device)
         (only-in torch/scribblings/examples/17-workers collate-digits encoded-digits)
         (only-in torch/vision/mnist load-mnist load-mnist-fixture)
         (only-in torch/vision/transforms random-crop))

(provide jittered
         training-set)

(define-dataset jittered (images labels)
  #:length (numel labels)
  #:ref (i)
  (values (~> images (select 0 i) (unsqueeze 0) (random-crop #:padding 2) (select 0 0))
          (select labels 0 i)))

(define (training-set variant dataset)
  (define-values (xs ys)
    (if (equal? dataset "fixture") (load-mnist-fixture) (load-mnist 'train)))
  (define-values (test-x test-y)
    (if (equal? dataset "fixture") (values xs ys) (load-mnist 'test)))
  (values (if (equal? variant "crop") (jittered xs ys) (encoded-digits xs ys))
          (if (equal? variant "crop") #f collate-digits)
          test-x
          test-y))

(module+ main
  (require (only-in "../../../examples/test/private/env.rkt" env-number))

  (define epochs (env-number "EPOCHS" 2))
  (define workers (env-number "WORKERS" 0))
  (define batch (env-number "BATCH" 64))
  (define variant (or (getenv "VARIANT") "pgm"))
  (define dataset (or (getenv "DATASET") "mnist"))
  (define device (pick-device))

  (define-values (ds collate test-x test-y) (training-set variant dataset))

  (define (loader generator)
    (if collate
        (dataloader ds #:batch-size batch #:shuffle? #t #:collate collate
                    #:generator generator #:workers workers)
        (dataloader ds #:batch-size batch #:shuffle? #t
                    #:generator generator #:workers workers)))

  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed 0)
    (with-default-device device
      (manual-seed! 0)
      (define net (convnet))
      (define opt (adam (parameters net) #:lr 0.001))
      (define xs (to test-x device))
      (define ys (to test-y device))
      (define batches (loader (make-generator 0)))
      (for ([_ (in-range epochs)])
        (for ([(xb yb) (in-dataloader batches)])
          (zero-grads! opt)
          (define loss (~> xb (to device) net (cross-entropy (to yb device))))
          (backward! loss)
          (step! opt)
          (item loss))
        (accuracy net xs ys)))))
