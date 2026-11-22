#lang info

#|review: ignore|#

(define c1-legs
  '("ema-state-test.rkt"
    "grad-scaler-state-parity-test.rkt"
    "grad-scaler-state-test.rkt"
    "optim-state-dict-parity-test.rkt"
    "optim-state-dict-test.rkt"
    "scheduler-state-parity-test.rkt"
    "scheduler-state-test.rkt"))

(define c2-legs
  '("checkpoint-codec-test.rkt"
    "checkpoint-format-parity-test.rkt"
    "checkpoint-format-test.rkt"
    "checkpoint-load-test.rkt"
    "trainer-hook-order-save-test.rkt"))

(define test-omit-paths (append c1-legs c2-legs))
