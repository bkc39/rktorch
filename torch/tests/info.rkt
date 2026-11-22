#lang info

#|review: ignore|#

(define c4-legs
  '("model-checkpoint-test.rkt"
    "on-exception-checkpoint-test.rkt"
    "trainer-ckpt-alias-test.rkt"
    "trainer-resume-test.rkt"))

(define c5-legs
  '("trainer-hook-order-resume-mid-test.rkt"
    "trainer-resume-mid-test.rkt"))

(define c6-legs
  '("trainer-resume-cuda-test.rkt"
    "trainer-resume-generator-test.rkt"))

(define test-omit-paths (append c4-legs c5-legs c6-legs))
