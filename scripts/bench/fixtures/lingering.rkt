#lang racket/base

(module+ main
  (require (submod "stepper.rkt" main)
           (only-in "grandchild.rkt" start-grandchild!))
  (start-grandchild!))
