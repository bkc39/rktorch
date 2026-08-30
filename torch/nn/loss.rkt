#lang racket/base

(require (only-in racket/contract ->* and/c listof)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../foreign.rkt"
                  device-type mean mul sub tensor-device tensor? to-device
                  to-dtype)
         (only-in "../generated.rkt"
                  [cross-entropy-loss g:cross-entropy-loss]
                  [ctc-loss-intlist g:ctc-loss-intlist]))

(provide mse-loss
         cross-entropy)

(define (mse-loss prediction target)
  (define d (sub prediction target))
  (mean (mul d d)))

(define (cross-entropy logits targets)
  (g:cross-entropy-loss logits (to-dtype targets 'int64) #f 1 -100 0.0))

(define input-lengths/c (and/c (listof exact-positive-integer?) pair?))
;; a 0 target length is a valid empty transcript
(define target-lengths/c (and/c (listof exact-nonnegative-integer?) pair?))

;; log-probs is (T N C) log-softmaxed frames; targets is (N S) labels.
;; Reduction is fixed to torch's mean like cross-entropy above.
(define/contract-out (ctc-loss log-probs targets
                           #:input-lengths input-lengths
                           #:target-lengths target-lengths
                           #:blank [blank 0]
                           #:zero-infinity? [zero-infinity? #f])
  (->* (tensor? tensor?
        #:input-lengths input-lengths/c
        #:target-lengths target-lengths/c)
       (#:blank exact-nonnegative-integer?
        #:zero-infinity? boolean?)
       tensor?)
  (define (marginalize lp tg)
    (g:ctc-loss-intlist lp (to-dtype tg 'int64)
                        input-lengths target-lengths blank 1 zero-infinity?))
  (define device (tensor-device log-probs))
  ;; aten::_ctc_loss is CPU/CUDA only, so MPS detours through the CPU;
  ;; to-device is differentiable both ways, so the gradient returns
  (if (eq? (device-type device) 'mps)
      (to-device (marginalize (to-device log-probs 'cpu)
                              (to-device targets 'cpu))
                 device)
      (marginalize log-probs targets)))
