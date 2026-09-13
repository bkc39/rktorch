#lang racket/base

(require (only-in racket/contract/base ->*)
         (only-in "../foreign.rkt" group-norm ones zeros)
         (only-in "layer.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer GroupNorm (num-groups eps weight bias) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:eps real?]
                  group-norm?)
  #:init (num-groups num-channels #:eps [eps 1e-5])
  (set! weight (Parameter (ones num-channels)))
  (set! bias (Parameter (zeros num-channels)))
  #:forward (x)
  (group-norm x num-groups #:weight weight #:bias bias #:eps eps))
