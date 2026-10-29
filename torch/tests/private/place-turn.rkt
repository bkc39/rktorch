#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in "../../foreign.rkt" zeros)
         (only-in "../../foreign/raw/collector.rkt" on-main-os-thread?))

(provide main)

(define (yields-after-a-collection?)
  (collect-garbage)
  (define ran? (box #f))
  (thread (lambda () (set-box! ran? #t)))
  (zeros 1)
  (unbox ran?))

(define (main ch)
  (place-channel-put ch (list (on-main-os-thread?)
                              (for/and ([_ (in-range 5)])
                                (yields-after-a-collection?)))))
