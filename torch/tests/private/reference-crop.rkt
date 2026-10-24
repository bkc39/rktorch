#lang racket/base

(require (only-in "../../foreign.rkt"
                  copy! device draw-seed dtype narrow select stack tensor-shape
                  zeros))

(provide reference-crop)

(define (batch-rng generator)
  (define rng (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator rng])
    (random-seed (bitwise-and (draw-seed #:generator generator) #x7FFFFFFF)))
  rng)

(define (reference-crop x #:padding [padding 4] #:generator [generator #f])
  (define dims (tensor-shape x))
  (define n (car dims))
  (define h (caddr dims))
  (define w (cadddr dims))
  (define padded (zeros n (cadr dims) (+ h (* 2 padding)) (+ w (* 2 padding))
                        #:device (device x)
                        #:dtype (dtype x)))
  (copy! (narrow (narrow padded 2 padding h) 3 padding w) x)
  (define rng (batch-rng generator))
  (define span (add1 (* 2 padding)))
  (stack (for/list ([i (in-range n)])
           (define top (random span rng))
           (define left (random span rng))
           (narrow (narrow (select padded 0 i) 1 top h) 2 left w))
         0))
