#lang racket/base

;; Runner + tests for the literate ../../examples/racket/09-resnet.rkt.

(require (except-in racket/list argmax flatten take)
         (only-in racket/math nan?)
         torch
         torch/nn
         (only-in torch/vision/cifar10 load-cifar10-fixture)
         "../racket/09-resnet.rkt")

(module+ main
  ;; The headline run: full CIFAR-10 (downloads + caches the 163 MB archive
  ;; once), test accuracy after every epoch. Pass EPOCHS to override.
  (define epochs (string->number (or (getenv "EPOCHS") "30")))
  (printf "device: ~a\n" (pick-device))
  (for ([acc (in-list (train-cifar10 #:epochs epochs))]
        [epoch (in-naturals 1)])
    (printf "epoch ~a: test acc ~a\n" epoch acc)
    (flush-output)))

(module+ test
  (require rackunit)
  ;; Deterministic, offline: two steps on eight fixture images.
  (define-values (losses net device)
    (run-example #:device 'cpu #:batch 8 #:steps 2))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 2)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (define names (map car (named-parameters net)))
  (check-equal? (take names 3) '("stem.weight" "bn.weight" "bn.bias"))
  (check-equal? (list-ref names 3) "layer1.0.conv1.weight")
  (check-equal? (last names) "fc.bias")
  ;; the stem and its norm, four stages of two blocks with six tensors
  ;; each and nine where the first block projects, then the head
  (check-equal? (length names) (+ 3 (* 2 6) (* 3 (+ 9 6)) 2))
  (check-equal? (tensor-shape (car (parameters net))) '(16 3 3 3))
  (check-not-false (member "layer2.0.shortcut.0.weight" names) "the projection")
  (check-false (member "layer1.0.shortcut.0.weight" names) "identity elsewhere")
  (check-equal? (tensor-shape (net (zeros 2 3 32 32))) '(2 10))
  (check-true (layer-training? net))
  (define-values (xs ys) (load-cifar10-fixture))
  (define acc (accuracy net xs ys))
  (check-true (<= 0.0 acc 1.0))
  (check-true (layer-training? net) "accuracy left the net in eval mode")
  ;; Device RNG streams differ from the CPU's for the init, so the on-device
  ;; arm checks convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net _a-dev)
      (run-example #:device accel #:batch 8 #:steps 2))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))))
