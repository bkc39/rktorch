#lang racket/base

;; Runner + tests for the literate ../../examples/racket/05-mnist.rkt.

(require (except-in racket/list argmax flatten take)
         (only-in racket/math nan?)
         torch
         torch/nn
         (only-in torch/data/mnist load-mnist-fixture)
         "../racket/05-mnist.rkt")

(module+ main
  ;; The headline run: full MNIST (downloads + caches), ~98% on a GPU. Pass
  ;; EPOCHS to override. Use run-example for the quick offline smoke instead.
  (define epochs (string->number (or (getenv "EPOCHS") "3")))
  (printf "device: ~a\n" (pick-device))
  (for ([acc (in-list (train-mnist #:epochs epochs))]
        [epoch (in-naturals 1)])
    (printf "epoch ~a: test acc ~a\n" epoch acc)))

(module+ test
  (require rackunit)
  ;; Deterministic, offline: 5 full-batch steps on the committed fixture.
  (define-values (losses net device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  ;; the convnet's parameter tree: conv/linear weights+biases in decl order.
  (check-equal? (map car (named-parameters net))
                '("c1.weight" "c1.bias" "c2.weight" "c2.bias"
                  "f1.weight" "f1.bias" "f2.weight" "f2.bias"))
  (check-equal? (map tensor-shape (parameters net))
                '((16 1 3 3) (16) (32 16 3 3) (32)
                  (128 800) (128) (10 128) (10)))
  ;; smoke-test accuracy (the only exerciser of in-eval-mode at the example
  ;; level): the loss-decreasing net should beat the 0.1 random-chance floor for
  ;; 10 classes, and accuracy must leave the net back in train mode.
  (define-values (xs ys) (load-mnist-fixture))
  (define acc (accuracy net xs ys))
  (check-true (and (> acc 0.1) (<= acc 1.0))
              (format "accuracy out of expected range: ~a" acc))
  (check-true (module-training? net) "accuracy left the net in eval mode")
  ;; Device RNG streams differ from the CPU's, so the on-device arm checks
  ;; convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net _a-dev) (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))))
