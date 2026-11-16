#lang racket/base

(require (only-in racket/place place-channel-get place-channel-put)
         (only-in "switches.rkt" set-switches! switch-values))

(provide die main)

(define (main ch)
  (place-channel-put ch (switch-values))
  (set-switches! (place-channel-get ch))
  (place-channel-put ch 'written))

(define (die _)
  (exit 3))
