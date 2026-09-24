#lang racket/base

;; Runner + tests for the literate ../../examples/racket/11-vae.rkt.

(require (except-in racket/list argmax flatten take)
         (only-in racket/math nan?)
         torch
         torch/nn
         "../racket/11-vae.rkt")

(module+ main
  ;; The headline run: full MNIST, mean loss per epoch, a 10x10 grid of
  ;; decoded latents per epoch under OUT. Pass EPOCHS to override.
  (define epochs (string->number (or (getenv "EPOCHS") "10")))
  (define out (getenv "OUT"))
  (printf "device: ~a\n" (pick-device))
  (for ([loss (in-list (train-vae #:epochs epochs #:out out))]
        [epoch (in-naturals 1)])
    (printf "epoch ~a: loss ~a\n" epoch loss)
    (flush-output)))

(module+ test
  (require rackunit)
  ;; Deterministic, offline: three steps on thirty-two fixture images.
  (define-values (losses net device)
    (run-example #:device 'cpu #:batch 32 #:steps 3))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 3)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (check-equal? (map car (named-parameters net))
                '("enc.weight" "enc.bias" "mu-head.weight" "mu-head.bias"
                  "logvar-head.weight" "logvar-head.bias"
                  "dec1.weight" "dec1.bias" "dec2.weight" "dec2.bias"))
  (define-values (logits mu logvar) (net (randn 4 1 28 28) (randn 4 20)))
  (check-equal? (tensor-shape logits) '(4 784))
  (check-equal? (tensor-shape mu) '(4 20))
  (check-equal? (tensor-shape logvar) '(4 20))
  (check-equal? (tensor-shape (decode net (randn 3 20))) '(3 784))
  (check-true (rational? (item (vae-loss logits (randn 4 1 28 28) mu logvar))))
  ;; Device RNG streams differ from the CPU's for the init, so the on-device
  ;; arm checks convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net _a-dev)
      (run-example #:device accel #:batch 32 #:steps 3))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))))
