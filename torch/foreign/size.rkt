#lang racket/base

;; Size-argument normalization for the conv/pool surface: an int
;; broadcasts to a square, an explicit [h w] list passes through,
;; mirroring PyTorch. Shared so promoted.rkt and nn/conv.rkt can't drift.

(provide ->2d)

(define (->2d x) (if (list? x) x (list x x)))
