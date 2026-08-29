#lang racket/base

(provide ->1d ->2d)

(define (->1d x) (if (list? x) x (list x)))

(define (->2d x) (if (list? x) x (list x x)))
