#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i non-empty-listof or/c unsupplied-arg?)
         (prefix-in g: (only-in "../generated.rkt"
                                adaptive-avg-pool2d
                                avg-pool2d
                                clamp
                                conv-transpose2d-input
                                conv1d
                                conv2d
                                embedding
                                group-norm
                                layer-norm
                                masked-fill-scalar
                                max-pool2d
                                repeat-interleave-self-int
                                silu
                                tril
                                triu))
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "contracts.rkt"
                  image-batch/c index/c nonneg-size-1d/c nonneg-size/c
                  pool-size/c pos-size-1d/c pos-size/c)
         (only-in "size.rkt" ->1d ->2d)
         (only-in "structs.rkt" tensor?))

(define/contract-out (conv1d input weight ;; noqa
                             #:bias [bias #f]
                             #:stride [stride 1]
                             #:padding [padding 0]
                             #:dilation [dilation 1]
                             #:groups [groups 1])
  (->* [tensor? tensor?]
       [#:bias (or/c tensor? #f) #:stride pos-size-1d/c
        #:padding nonneg-size-1d/c #:dilation pos-size-1d/c
        #:groups exact-positive-integer?]
       tensor?)
  (g:conv1d input weight bias
            (->1d stride) (->1d padding) (->1d dilation) groups))

(define/contract-out (conv2d input weight ;; noqa
                             #:bias [bias #f]
                             #:stride [stride 1]
                             #:padding [padding 0]
                             #:dilation [dilation 1]
                             #:groups [groups 1])
  (->* [tensor? tensor?]
       [#:bias (or/c tensor? #f) #:stride pool-size/c
        #:padding pool-size/c #:dilation pool-size/c
        #:groups index/c]
       tensor?)
  (g:conv2d input weight bias
            (->2d stride) (->2d padding) (->2d dilation) groups))

(define/contract-out (max-pool2d input kernel-size ;; noqa
                                 #:stride [stride #f]
                                 #:padding [padding 0]
                                 #:dilation [dilation 1]
                                 #:ceil-mode [ceil-mode #f])
  (->* [tensor? pool-size/c]
       [#:stride (or/c pool-size/c #f) #:padding pool-size/c
        #:dilation pool-size/c #:ceil-mode boolean?]
       tensor?)
  (g:max-pool2d input
                (->2d kernel-size)
                (->2d (or stride kernel-size))
                (->2d padding) (->2d dilation) ceil-mode))

(define/contract-out (avg-pool2d input kernel-size ;; noqa
                                 #:stride [stride #f]
                                 #:padding [padding 0]
                                 #:ceil-mode [ceil-mode #f]
                                 #:count-include-pad [count-include-pad #t]
                                 #:divisor-override [divisor-override #f])
  (->* [tensor? pool-size/c]
       [#:stride (or/c pool-size/c #f) #:padding pool-size/c
        #:ceil-mode boolean? #:count-include-pad boolean?
        #:divisor-override (or/c exact-positive-integer? #f)]
       tensor?)
  (g:avg-pool2d input
                (->2d kernel-size)
                (->2d (or stride kernel-size))
                (->2d padding) ceil-mode count-include-pad divisor-override))

(define/contract-out (adaptive-avg-pool2d input output-size) ;; noqa
  (-> tensor? pool-size/c tensor?)
  (g:adaptive-avg-pool2d input (->2d output-size)))

(define/contract-out (tril self [diagonal 0]) ;; noqa
  (->* [tensor?] [exact-integer?] tensor?)
  (g:tril self diagonal))

(define/contract-out (triu self [diagonal 0]) ;; noqa
  (->* [tensor?] [exact-integer?] tensor?)
  (g:triu self diagonal))

(define/contract-out (masked-fill self mask value) ;; noqa
  (-> tensor? tensor? real? tensor?)
  (g:masked-fill-scalar self mask (exact->inexact value)))

(define/contract-out (embedding indices weight #:padding-idx [padding-idx #f]) ;; noqa
  (->* [tensor? tensor?]
       [#:padding-idx (or/c #f exact-nonnegative-integer?)]
       tensor?)
  (g:embedding weight indices (or padding-idx -1) #f #f))

(define/contract-out (layer-norm input normalized-shape ;; noqa
                                 #:weight [weight #f]
                                 #:bias [bias #f]
                                 #:eps [eps 1e-5])
  (->* [tensor?
        (or/c exact-positive-integer?
              (non-empty-listof exact-positive-integer?))]
       [#:weight (or/c tensor? #f)
        #:bias (or/c tensor? #f)
        #:eps real?]
       tensor?)
  (define shape
    (if (list? normalized-shape) normalized-shape (list normalized-shape)))
  (g:layer-norm input shape weight bias eps #t))

(define/contract-out (conv-transpose2d input weight ;; noqa
                                       #:bias [bias #f]
                                       #:stride [stride 1]
                                       #:padding [padding 0]
                                       #:output-padding [output-padding 0]
                                       #:dilation [dilation 1]
                                       #:groups [groups 1])
  (->* [tensor? tensor?]
       [#:bias (or/c tensor? #f) #:stride pos-size/c
        #:padding nonneg-size/c #:output-padding nonneg-size/c
        #:dilation pos-size/c #:groups exact-positive-integer?]
       tensor?)
  (g:conv-transpose2d-input input weight bias
                            (->2d stride) (->2d padding) (->2d output-padding)
                            groups (->2d dilation)))

(define/contract-out (group-norm input num-groups ;; noqa
                                 #:weight [weight #f]
                                 #:bias [bias #f]
                                 #:eps [eps 1e-5])
  (->* [tensor? exact-positive-integer?]
       [#:weight (or/c tensor? #f) #:bias (or/c tensor? #f) #:eps real?]
       tensor?)
  (g:group-norm input num-groups weight bias (exact->inexact eps) #t))

(define/contract-out silu (-> tensor? tensor?) g:silu) ;; noqa

(define/contract-out (clamp self #:min [min #f] #:max [max #f]) ;; noqa
  (->i ([self tensor?])
       (#:min [min (or/c real? #f)] #:max [max (or/c real? #f)])
       #:pre/name (min max)
       "at least one bound"
       (or (and (not (unsupplied-arg? min)) min)
           (and (not (unsupplied-arg? max)) max))
       [result tensor?])
  (g:clamp self (and min (exact->inexact min)) (and max (exact->inexact max))))

(define/contract-out (upsample-nearest2d input #:scale [scale 2]) ;; noqa
  (->* [image-batch/c] [#:scale exact-positive-integer?] tensor?)
  (g:repeat-interleave-self-int
   (g:repeat-interleave-self-int input scale 2 #f) scale 3 #f))
