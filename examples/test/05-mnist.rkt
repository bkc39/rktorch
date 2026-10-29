#lang racket/base

;; Runner + tests for the literate torch/scribblings/examples/05-mnist.rkt.

(require (except-in racket/list argmax flatten take)
         (only-in racket/math nan?)
         torch
         torch/data/loader
         torch/nn
         torch/scribblings/examples/05-mnist
         (only-in torch/train fit run-metric)
         (only-in torch/vision/mnist load-mnist-fixture))

(module+ main
  (require (only-in "private/env.rkt" env-number))
  ;; The headline run: full MNIST (downloads + caches), ~98% on a GPU. Pass
  ;; EPOCHS to override. Use run-example for the quick offline smoke instead.
  (define epochs (env-number "EPOCHS" 3))
  (printf "device: ~a\n" (pick-device))
  (printf "test acc ~a\n" (train-mnist #:epochs epochs)))

(module+ test
  (require rackunit)
  ;; Deterministic, offline: 5 full-batch steps on the committed fixture.
  (define-values (losses classifier trained-on) (run-example #:device 'cpu))
  (check-equal? trained-on 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (define hand-losses
    (with-default-device 'cpu
      (manual-seed! 0)
      (define-values (xs ys) (load-mnist-fixture))
      (define net (convnet))
      (define opt (adam (parameters net) #:lr 0.001))
      (for/list ([_ (in-range 5)])
        (zero-grads! opt)
        (define loss (cross-entropy (net xs) ys))
        (backward! loss)
        (step! opt)
        (item loss))))
  (check-equal? losses hand-losses)
  (check-equal? (map car (named-parameters classifier))
                '("net.0.conv.weight" "net.0.conv.bias"
                  "net.1.conv.weight" "net.1.conv.bias"
                  "net.3.weight" "net.3.bias" "net.5.weight" "net.5.bias"))
  (check-equal? (map tensor-shape (parameters classifier))
                '((16 1 3 3) (16) (32 16 3 3) (32)
                  (128 800) (128) (10 128) (10)))
  (define-values (xs ys) (load-mnist-fixture))
  (define acc (accuracy classifier xs ys))
  (check-true (and (> acc 0.1) (<= acc 1.0))
              (format "accuracy out of expected range: ~a" acc))
  (check-true (layer-training? classifier)
              "accuracy left the classifier in eval mode")
  (define fixture (dataloader (tensor-dataset xs ys) #:batch-size 64))
  (define r (fit classifier fixture #:val fixture #:max-epochs 1 #:logger #f
                 #:num-sanity-val-steps 0))
  (check-= (run-metric r 'val/acc) (accuracy classifier xs ys) 1e-6)
  ;; Device RNG streams differ from the CPU's, so the on-device arm checks
  ;; convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-classifier _a-dev)
      (run-example #:device accel))
    (check-equal? (device (car (parameters a-classifier))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))))
