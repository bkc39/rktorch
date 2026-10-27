#lang racket/base

(provide adam-foreach!
         adam-fused!
         rmsprop-foreach!
         sgd-foreach!
         sgd-fused!)

(define (sgd-foreach! state ps gs #:lr lr #:momentum momentum
                      #:nesterov? nesterov? #:weight-decay weight-decay)
  (error 'unimplemented))

(define (sgd-fused! state ps gs #:lr lr #:momentum momentum
                    #:nesterov? nesterov? #:weight-decay weight-decay)
  (error 'unimplemented))

(define (adam-foreach! state ps gs #:lr lr #:beta1 beta1 #:beta2 beta2
                       #:eps eps #:weight-decay weight-decay
                       #:decoupled? decoupled?)
  (error 'unimplemented))

(define (adam-fused! state ps gs #:lr lr #:beta1 beta1 #:beta2 beta2
                     #:eps eps #:weight-decay weight-decay
                     #:decoupled? decoupled?)
  (error 'unimplemented))

(define (rmsprop-foreach! state ps gs #:lr lr #:alpha alpha #:eps eps
                          #:weight-decay weight-decay #:momentum momentum)
  (error 'unimplemented))
