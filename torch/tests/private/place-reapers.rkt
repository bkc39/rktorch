#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in "../../foreign.rkt" + sum zeros))

(provide main)

(define (churn!)
  (for ([_ (in-range 2000)])
    (sum (+ (zeros 16 16) 1.0))))

(define (main ch)
  (define workers
    (for/list ([_ (in-range 2)])
      (thread churn! #:pool 'own)))
  (for-each thread-wait workers)
  (collect-garbage)
  (zeros 1)
  (place-channel-put ch 'churned))
