#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in "../../foreign.rkt" zeros)
         (only-in "../../foreign/raw/collector.rkt" on-main-os-thread?)
         (only-in (submod "../../runtime/reaper.rkt" private) reaper-statistics))

(provide main)

(define (wakes)
  (hash-ref (reaper-statistics) 'wakes))

(define (wakes-a-reaper-after-a-collection?)
  (sleep 0.02)
  (define before (wakes))
  (collect-garbage)
  (zeros 1)
  (define deadline (+ (current-inexact-milliseconds) 1000))
  (let spin ()
    (cond
      [(> (wakes) before) #t]
      [(> (current-inexact-milliseconds) deadline) #f]
      [else (spin)])))

(define (main ch)
  (place-channel-put ch (list (on-main-os-thread?)
                              (for/and ([_ (in-range 5)])
                                (wakes-a-reaper-after-a-collection?)))))
