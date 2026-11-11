#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in "../../foreign.rkt" + sum zeros))

(provide main)

(define (main ch)
  (for ([_ (in-range 50)])
    (sum (+ (zeros 16 16) 1.0)))
  (collect-garbage)
  (place-channel-put ch 'made))
