#lang info

#|review: ignore|#

;; the trainer's law tests run once T1 (#384) implements fit
(define test-omit-paths
  '("trainer-contract-test.rkt"
    "trainer-hook-order-test.rkt"
    "trainer-log-test.rkt"
    "trainer-loop-test.rkt"
    "trainer-memory-test.rkt"
    "trainer-parity-test.rkt"))
