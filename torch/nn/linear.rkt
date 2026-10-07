#lang racket/base

(require (only-in racket/contract/base ->*)
         (only-in "../foreign.rkt" linear)
         (only-in "init.rkt" kaiming-uniform uniform-bias)
         (only-in "layer.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer Linear (weight bias) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:bias? boolean?]
                  linear?)
  #:init (in-features out-features #:bias? [bias? #t] #:drawn [drawn #f])
  ;; weight before bias: nn.Linear.reset_parameters' RNG draw order
  (set! weight
        (Parameter (if drawn
                       (car drawn)
                       (kaiming-uniform (list out-features in-features)))))
  (set! bias
        (and bias?
             (Parameter (if drawn
                            (cdr drawn)
                            (uniform-bias out-features in-features)))))
  #:forward (x)
  (linear x weight #:bias bias))

;; A layer that splits one drawn tensor across several projections, as
;; MultiheadAttention does nn.MultiheadAttention's in_proj_weight, needs a
;; Linear that draws nothing; #:drawn is outside the exported contract.
(module+ private
  (require (only-in racket/match match-define)
           (only-in "../foreign.rkt" shape))
  (provide tensors->Linear) ;; noqa
  (define (tensors->Linear weight bias)
    (match-define (list out-features in-features) (shape weight))
    (Linear in-features out-features
            #:bias? (and bias #t)
            #:drawn (cons weight bias))))
