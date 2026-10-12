#lang racket/base

(require (only-in racket/match match)
         (only-in "../foreign.rkt"
                  device-type flatten item select tensor-device with-no-grad))

(provide log-optimizer-step!)

(define-logger rktorch-step)

(define (settle! params)
  (match params
    [(cons p _)
     #:when (not (eq? (device-type (tensor-device p)) 'cpu))
     (with-no-grad (item (select (flatten p) 0 0)))]
    [_ (void)]))

(define (log-optimizer-step! opt params)
  (when (log-level? rktorch-step-logger 'debug 'rktorch-step)
    (settle! params)
    (log-message rktorch-step-logger 'debug 'rktorch-step "step"
                 (vector (eq-hash-code opt)
                         (current-inexact-monotonic-milliseconds)
                         (current-gc-milliseconds))
                 #f)))
