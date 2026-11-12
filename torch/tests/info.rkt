#lang info

#|review: ignore|#

(define t2-legs
  '("grad-scaler-parity-test.rkt"
    "grad-scaler-test.rkt"
    "optim-param-groups-parity-test.rkt"
    "optim-param-groups-test.rkt"
    "scheduler-chain-test.rkt"
    "scheduler-parity-test.rkt"
    "trainer-accumulation-test.rkt"
    "trainer-clip-test.rkt"
    "trainer-hparam-test.rkt"
    "trainer-manual-test.rkt"
    "trainer-precision-test.rkt"
    "trainer-schedule-test.rkt"))

(define test-omit-paths t2-legs)
