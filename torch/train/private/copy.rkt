#lang racket/base

(require (only-in "../../foreign.rkt" copy! with-no-grad zeros-like))

(provide copy-of)

(define (copy-of t)
  (define copy (zeros-like t))
  (with-no-grad (copy! copy t))
  copy)
