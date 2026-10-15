#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in "../foreign.rkt"
                  device-type flatten item numel select tensor-device with-no-grad))

(provide log-optimizer-step!
         settle!)

(define-logger rktorch-step)

(define (accelerator-of p)
  (define d (tensor-device p))
  (and (not (eq? (device-type d) 'cpu)) d))

(define (settle! params #:accelerator-of [accelerator-of accelerator-of])
  (define on-accelerators
    (for*/list ([p (in-list params)]
                #:when (positive? (numel p))
                [d (in-value (accelerator-of p))]
                #:when d)
      (cons d p)))
  (define synced (map cdr (remove-duplicates on-accelerators #:key car)))
  (for ([p (in-list synced)])
    (with-no-grad (item (select (flatten p) 0 0))))
  synced)

(define (log-optimizer-step! opt params)
  (when (log-level? rktorch-step-logger 'debug 'rktorch-step)
    (settle! params)
    (log-message rktorch-step-logger 'debug 'rktorch-step "step"
                 (vector (eq-hash-code opt)
                         (current-inexact-monotonic-milliseconds)
                         (current-gc-milliseconds))
                 #f)))
