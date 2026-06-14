#lang racket/base

;; Shared size-argument normalization for the conv/pool surface. A kernel /
;; stride / padding arg is either an int (broadcast to a square) or an
;; explicit [h w] list, mirroring PyTorch. Lives in one place so the
;; functional wrappers (foreign/promoted.rkt) and the nn layers
;; (nn/conv.rkt) can't drift.

(provide ->2d)

(define (->2d x) (if (list? x) x (list x x)))
