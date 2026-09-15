#lang racket/base

(require (only-in racket/contract/base ->i)
         (only-in "../foreign.rkt"
                  device-type group-norm ones tensor-device to-device zeros)
         (only-in "layer.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer GroupNorm (num-groups eps weight bias) ;; noqa
  #:contract (->i ([num-groups exact-positive-integer?]
                   [num-channels exact-positive-integer?])
                  (#:eps [eps real?])
                  #:pre/name (num-groups num-channels)
                  "the groups must divide the channels"
                  (zero? (remainder num-channels num-groups))
                  [result group-norm?])
  #:init (num-groups num-channels #:eps [eps 1e-5])
  (set! weight (Parameter (ones num-channels)))
  (set! bias (Parameter (zeros num-channels)))
  #:forward (x)
  ;; libtorch 2.9 has no MPS kernel for the backward, so MPS detours through
  ;; the CPU; to-device is differentiable both ways, so the gradient returns
  (define device (tensor-device x))
  (if (eq? (device-type device) 'mps)
      (to-device (group-norm (to-device x 'cpu) num-groups
                             #:weight (to-device weight 'cpu)
                             #:bias (to-device bias 'cpu)
                             #:eps eps)
                 device)
      (group-norm x num-groups #:weight weight #:bias bias #:eps eps)))
