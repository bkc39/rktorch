#lang racket/base

(require (only-in racket/contract
                  ->* and/c define/contract listof)
         (only-in "../foreign.rkt" mean mul sub tensor? to-dtype)
         (only-in "../generated.rkt"
                  [cross-entropy-loss g:cross-entropy-loss]
                  [ctc-loss-intlist g:ctc-loss-intlist]))

(provide mse-loss
         cross-entropy
         ctc-loss)

(define (mse-loss prediction target)
  (define d (sub prediction target))
  (mean (mul d d)))

(define (cross-entropy logits targets)
  (g:cross-entropy-loss logits (to-dtype targets 'int64) #f 1 -100 0.0))

(define lengths/c (and/c (listof exact-positive-integer?) pair?))

;; log-probs is (T N C) log-softmaxed frames; targets is (N S) labels.
;; Reduction is fixed to torch's mean like cross-entropy above.
(define/contract (ctc-loss log-probs targets
                           #:input-lengths input-lengths
                           #:target-lengths target-lengths
                           #:blank [blank 0]
                           #:zero-infinity? [zero-infinity? #f])
  (->* (tensor? tensor?
        #:input-lengths lengths/c
        #:target-lengths lengths/c)
       (#:blank exact-nonnegative-integer?
        #:zero-infinity? boolean?)
       tensor?)
  (g:ctc-loss-intlist log-probs (to-dtype targets 'int64)
                      input-lengths target-lengths blank 1 zero-infinity?))
