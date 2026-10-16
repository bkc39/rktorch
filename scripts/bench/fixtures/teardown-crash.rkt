#lang racket/base

(module+ main
  (require (submod "stepper.rkt" main))
  (exit-handler (let ([exit-with (exit-handler)]) (lambda (_code) (exit-with 139)))))
