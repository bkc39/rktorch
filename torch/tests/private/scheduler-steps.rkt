#lang racket/base

(require (only-in "../../nn.rkt" step!))

(provide stepped-reads)

(define (stepped-reads s read steps)
  (cons (read s) (for/list ([_ (in-range steps)]) (step! s) (read s))))
