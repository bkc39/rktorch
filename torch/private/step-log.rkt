#lang racket/base

(require (only-in racket/match match)
         (only-in "../foreign.rkt"
                  device-type flatten item numel select tensor-device with-no-grad))

(provide log-optimizer-step!
         settle!)

(define-logger rktorch-step)

(define (accelerator? p)
  (not (eq? (device-type (tensor-device p)) 'cpu)))

(define (settle! params #:accelerator? [on-accelerator? accelerator?])
  (match (for/first ([p (in-list params)] #:when (positive? (numel p))) p)
    [#f (void)]
    [(? on-accelerator? p) (with-no-grad (item (select (flatten p) 0 0)))]
    [_ (void)]))

(define (log-optimizer-step! opt params)
  (when (log-level? rktorch-step-logger 'debug 'rktorch-step)
    (settle! params)
    (log-message rktorch-step-logger 'debug 'rktorch-step "step"
                 (vector (eq-hash-code opt)
                         (current-inexact-monotonic-milliseconds)
                         (current-gc-milliseconds))
                 #f)))
