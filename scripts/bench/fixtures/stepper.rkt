#lang racket/base

(module+ main
  (require (only-in torch * backward! ones sum ~>)
           (only-in torch/nn Parameter sgd step!)
           (only-in "../../../examples/test/private/env.rkt" env-number))
  (define epochs (env-number "EPOCHS" 2))
  (define p (Parameter (ones 2)))
  (define opt (sgd (list p) #:lr 0.01))
  (define step (if (getenv "BENCH_PREHOOK") void step!))
  (for ([_ (in-range (env-number "STEPS" (* 3 epochs)))])
    (~> (* p p) sum backward!)
    (step opt)))
