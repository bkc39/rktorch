#lang racket/base

(require (only-in racket/contract/base
                  ->* ->i flat-named-contract or/c unsupplied-arg?)
         (only-in "../foreign.rkt"
                  arange cat cos default-device device/c exp masked-fill mul
                  ones reshape sin stack tensor-device tensor-shape tensor?
                  to-dtype triu unsqueeze zeros)
         (only-in "../private/contract.rkt" define/contract-out))

(define even-width/c
  (flat-named-contract
   'even-width
   (lambda (n) (and (exact-positive-integer? n) (even? n)))))

(define position-vector/c
  (flat-named-contract
   'position-vector
   (lambda (t) (and (tensor? t) (= 1 (length (tensor-shape t)))))))

(define/contract-out (sinusoidal-positions positions width ;; noqa
                                           #:layout [layout 'interleaved]
                                           #:device [device (default-device)])
  (->i ([positions (or/c exact-nonnegative-integer? position-vector/c)]
        [width even-width/c])
       (#:layout [layout (or/c 'interleaved 'halves)]
        #:device [device device/c])
       #:pre/name (positions device)
       "#:device places a length; a tensor of positions keeps its own device"
       (or (unsupplied-arg? device) (exact-integer? positions))
       [_ tensor?])
  (define at
    (if (tensor? positions)
        (to-dtype positions 'float32)
        (arange positions #:device device)))
  (define half (quotient width 2))
  (define frequencies
    (exp (mul (arange half #:device (tensor-device at))
              (- (/ (log 10000.0) half)))))
  (define angles (mul (unsqueeze at 1) (unsqueeze frequencies 0)))
  (define waves (list (sin angles) (cos angles)))
  (if (eq? layout 'halves)
      (cat waves 1)
      (reshape (stack waves 2) (car (tensor-shape at)) width)))

(define/contract-out (causal-mask size ;; noqa
                                  #:device [device (default-device)]
                                  #:dtype [dtype 'bool])
  (->* [exact-nonnegative-integer?]
       [#:device device/c
        #:dtype (or/c 'bool 'float32 'float64 'float16 'bfloat16)]
       tensor?)
  (define later (triu (ones size size #:device device #:dtype 'bool) 1))
  (if (eq? dtype 'bool)
      later
      (masked-fill (zeros size size #:device device #:dtype dtype)
                   later
                   -inf.0)))
