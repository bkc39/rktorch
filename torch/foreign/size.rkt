#lang racket/base

(provide ->2d)

(define (->2d x) (if (list? x) x (list x x)))
