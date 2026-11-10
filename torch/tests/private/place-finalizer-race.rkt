#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in "finalizer-race.rkt" run-race))

(provide main)

(define (main ch)
  (place-channel-put ch (run-race 8 2000)))
