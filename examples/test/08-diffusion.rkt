#lang racket/base

;; Runner + tests for the literate ../../examples/racket/08-diffusion.rkt.

(require (except-in racket/list argmax flatten take)
         (only-in racket/math nan?)
         torch
         torch/nn
         "../racket/08-diffusion.rkt")

(module+ main
  ;; The headline run: full CIFAR-10 (downloads + caches the 163 MB archive
  ;; once), mean epsilon-MSE per epoch. Pass EPOCHS to override.
  (define epochs (string->number (or (getenv "EPOCHS") "10")))
  (printf "device: ~a\n" (pick-device))
  (for ([loss (in-list (train-cifar10 #:epochs epochs))]
        [epoch (in-naturals 1)])
    (printf "epoch ~a: mean loss ~a\n" epoch loss)))

(module+ test
  (require rackunit)
  (define-values (losses net device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (define names (map car (named-parameters net)))
  (check-equal? (take names 4)
                '("time.fc1.weight" "time.fc1.bias" "time.fc2.weight" "time.fc2.bias"))
  (check-equal? (last names) "out-conv.bias")
  (check-equal? (length names) 144)
  (check-equal? (tensor-shape (car (parameters net))) '(256 64))
  (check-equal? (tensor-shape (net (zeros 2 3 32 32) (tensor '(0 999) #:dtype 'int64) #f))
                '(2 3 32 32))
  ;; Device RNG streams differ from the CPU's for the init, so the on-device
  ;; arm checks convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net _a-dev) (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))))
