#lang racket/base

(require (only-in racket/match match)
         (only-in "../../foreign.rkt" cpu-device device-index device-type)
         (only-in "../../foreign/generator.rkt"
                  current-generator generator-device generator-state rng-state))

(provide rng-bundle)

(define (device-form d)
  (match (device-type d)
    ['cuda (list 'cuda (device-index d))]
    [type type]))

(define (rng-bundle [device (cpu-device)])
  (define streams
    (hash "cpu" (rng-state)
          "current_generator" (match (current-generator)
                                [#f #f]
                                [g (hash "device" (device-form (generator-device g))
                                         "state" (generator-state g))])
          "racket" (vector->list
                    (pseudo-random-generator->vector (current-pseudo-random-generator)))))
  (match (device-type device)
    ['cpu streams]
    [type (hash-set streams (symbol->string type) (rng-state #:device device))]))
