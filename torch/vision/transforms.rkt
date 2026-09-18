#lang racket/base

(require (only-in racket/contract/base ->* or/c real-in)
         (only-in "../foreign.rkt"
                  copy! draw-seed flip generator? narrow ne reshape select
                  stack tensor tensor-device tensor-dtype tensor-shape tensor?
                  where zeros)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../private/contract.rkt" define/contract-out))

;; One draw from the torch generator seeds a Racket generator for the batch,
;; so a seeded loader replays its augmentation while the images stay on the
;; device; random-seed takes 31 bits of the 63 drawn.
(define (batch-rng generator)
  (define rng (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator rng])
    (random-seed (bitwise-and (draw-seed #:generator generator) #x7FFFFFFF)))
  rng)

(define/contract-out (random-horizontal-flip x ;; noqa
                                             #:p [p 0.5]
                                             #:generator [generator #f])
  (->* [image-batch/c] [#:p (real-in 0 1) #:generator (or/c generator? #f)]
       tensor?)
  (define n (car (tensor-shape x)))
  (define rng (batch-rng generator))
  (define flipped
    (for/list ([_ (in-range n)]) (if (< (random rng) p) 1 0)))
  (define mask (reshape (ne (tensor flipped #:device (tensor-device x)) 0)
                        n 1 1 1))
  (where mask (flip x 3) x))

(define/contract-out (random-crop x ;; noqa
                                  #:padding [padding 4]
                                  #:generator [generator #f])
  (->* [image-batch/c]
       [#:padding exact-nonnegative-integer? #:generator (or/c generator? #f)]
       tensor?)
  (define dims (tensor-shape x))
  (define n (car dims))
  (define h (caddr dims))
  (define w (cadddr dims))
  (define padded (zeros n (cadr dims) (+ h (* 2 padding)) (+ w (* 2 padding))
                        #:device (tensor-device x)
                        #:dtype (tensor-dtype x)))
  (copy! (narrow (narrow padded 2 padding h) 3 padding w) x)
  (define rng (batch-rng generator))
  (define span (add1 (* 2 padding)))
  (stack (for/list ([i (in-range n)])
           (define top (random span rng))
           (define left (random span rng))
           (narrow (narrow (select padded 0 i) 1 top h) 2 left w))
         0))
