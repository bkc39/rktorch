#lang racket/base

(require (only-in racket/match match match*)
         (only-in threading ~>)
         (only-in "../../foreign.rkt" cpu-device device-index device-type)
         (only-in "../../foreign/generator.rkt"
                  current-generator generator-device generator-state rng-state
                  set-generator-state! set-rng-state!))

(provide check-current-generator!
         restore-rng-bundle!
         rng-bundle)

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

(define (check-current-generator! saved path)
  (define (refuse message . fields)
    (apply raise-arguments-error 'fit message (append fields (list "path" path))))
  (match* (saved (current-generator))
    [(#f _) (void)]
    [((hash-table ("device" form)) #f)
     (refuse "the checkpoint was saved under a current-generator, and this fit has none"
             "saved generator's device" form)]
    [((hash-table ("device" form)) g)
     (define here (~> g generator-device device-form))
     (unless (equal? here form)
       (refuse "the checkpoint's current-generator is on another device"
               "saved generator's device" form
               "current-generator's device" here))]))

(define (restore-current-generator! saved)
  (match* (saved (current-generator))
    [((hash-table ("state" state)) (? values g)) (set-generator-state! g state)]
    [(_ _) (void)]))

(define (restore-rng-bundle! bundle [device (cpu-device)])
  (define racket-state
    (~> bundle (hash-ref "racket") list->vector vector->pseudo-random-generator
        pseudo-random-generator->vector))
  (restore-current-generator! (hash-ref bundle "current_generator" #f))
  (set-rng-state! (hash-ref bundle "cpu"))
  (define type (device-type device))
  (define device-state (and (not (eq? type 'cpu)) (hash-ref bundle (symbol->string type) #f)))
  (when device-state
    (set-rng-state! device-state #:device device))
  (vector->pseudo-random-generator! (current-pseudo-random-generator) racket-state))
