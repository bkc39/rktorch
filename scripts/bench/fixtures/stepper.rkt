#lang racket/base

(module+ main
  (require (only-in torch backward! mul ones sum)
           (only-in torch/nn Parameter sgd step!))
  (define epochs (string->number (or (getenv "EPOCHS") "2")))
  (define p (Parameter (ones 2)))
  (define opt (sgd (list p) #:lr 0.01))
  (for ([_ (in-range (* 3 epochs))])
    (backward! (sum (mul p p)))
    (step! opt)))
