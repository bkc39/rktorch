#lang racket/base

(provide log-optimizer-step!)

(define-logger rktorch-step)

(define (log-optimizer-step! opt)
  (when (log-level? rktorch-step-logger 'debug)
    (log-message rktorch-step-logger 'debug 'rktorch-step "step"
                 (vector (eq-hash-code opt)
                         (current-inexact-monotonic-milliseconds)
                         (current-gc-milliseconds))
                 #f)))
