#lang racket/base

(require (only-in racket/match match-define)
         (only-in "../../foreign.rkt"
                  copy! device dtype item narrow select stack tensor-shape zeros)
         (only-in "../../generated.rkt" random-from!))

(provide reference-crop)

(define (reference-crop x #:padding [padding 4] #:generator [generator #f])
  (match-define (list n c h w) (tensor-shape x))
  (define padded (zeros n c (+ h (* 2 padding)) (+ w (* 2 padding))
                        #:device (device x)
                        #:dtype (dtype x)))
  (copy! (narrow (narrow padded 2 padding h) 3 padding w) x)
  (define span (add1 (* 2 padding)))
  (define (offset)
    (if (= span 1)
        0
        (item (random-from! (zeros 1 #:device 'cpu #:dtype 'int64)
                            0 span generator))))
  (stack (for/list ([i (in-range n)])
           (define top (offset))
           (define left (offset))
           (narrow (narrow (select padded 0 i) 1 top h) 2 left w))
         0))
