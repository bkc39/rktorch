#lang racket/base

;; Runner + tests for the literate ../../examples/racket/10-dcgan.rkt.

(require (except-in racket/list argmax flatten take)
         (only-in racket/math nan?)
         torch
         torch/nn
         "../racket/10-dcgan.rkt")

(module+ main
  ;; The headline run: full MNIST, mean discriminator and generator losses
  ;; per epoch, a 10x10 sample grid per epoch under OUT. Pass EPOCHS to
  ;; override.
  (define epochs (string->number (or (getenv "EPOCHS") "5")))
  (define out (getenv "OUT"))
  (printf "device: ~a\n" (pick-device))
  (for ([losses (in-list (train-dcgan #:epochs epochs #:out out))]
        [epoch (in-naturals 1)])
    (printf "epoch ~a: d ~a g ~a\n" epoch (car losses) (cadr losses))
    (flush-output)))

(module+ test
  (require rackunit)
  (define (finite? l) (and (rational? l) (not (nan? l))))
  ;; Deterministic, offline: two steps on sixteen fixture images.
  (define-values (d-losses g-losses gen disc device)
    (run-example #:device 'cpu #:batch 16 #:steps 2))
  (check-equal? device 'cpu)
  (check-equal? (length d-losses) 2)
  (check-equal? (length g-losses) 2)
  (check-true (andmap finite? (append d-losses g-losses))
              (format "non-finite loss: ~a ~a" d-losses g-losses))
  (check-equal? (map car (named-parameters gen))
                '("fc.weight" "fc.bias" "bn0.weight" "bn0.bias"
                  "up1.weight" "up1.bias" "bn1.weight" "bn1.bias"
                  "up2.weight" "up2.bias"))
  (check-equal? (map tensor-shape (parameters disc))
                '((64 1 4 4) (64) (128 64 4 4) (128) (128) (128) (1 6272) (1)))
  (check-equal? (tensor-shape (gen (randn 4 100))) '(4 1 28 28))
  (check-equal? (tensor-shape (disc (randn 4 1 28 28))) '(4 1))
  (define samples (in-eval-mode gen (with-no-grad (gen (randn 2 100)))))
  (check-true (for/and ([v (in-list (tensor->list samples))])
                (<= -1.0 v 1.0))
              "tanh keeps the images in [-1, 1]")
  (check-true (layer-training? gen))
  ;; Device RNG streams differ from the CPU's for the init, so the on-device
  ;; arm checks that both losses stay finite, never equality.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-d a-g a-gen _a-disc _a-dev)
      (run-example #:device accel #:batch 16 #:steps 2))
    (check-equal? (tensor-device (car (parameters a-gen))) accel)
    (check-true (andmap finite? (append a-d a-g))
                (format "non-finite loss on ~a: ~a ~a" accel a-d a-g))))
