#lang racket/base

(require (only-in racket/contract/base -> ->* non-empty-listof or/c)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "contracts.rkt"
                  index/c nonneg-size-1d/c pool-size/c pos-size-1d/c)
         (only-in "size.rkt" ->1d ->2d)
         (only-in "structs.rkt" tensor?)
         (prefix-in g: (only-in "../generated.rkt"
                                adaptive-avg-pool2d
                                avg-pool2d
                                conv1d
                                conv2d
                                embedding
                                layer-norm
                                masked-fill-scalar
                                max-pool2d
                                tril
                                triu)))

(define/contract-out (conv1d input weight
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

(define/contract-out (conv2d input weight
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

(define/contract-out (max-pool2d input kernel-size
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

(define/contract-out (avg-pool2d input kernel-size
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

(define/contract-out (adaptive-avg-pool2d input output-size)
  (-> tensor? pool-size/c tensor?)
  (g:adaptive-avg-pool2d input (->2d output-size)))

(define/contract-out (tril self [diagonal 0])
  (->* [tensor?] [exact-integer?] tensor?)
  (g:tril self diagonal))

(define/contract-out (triu self [diagonal 0])
  (->* [tensor?] [exact-integer?] tensor?)
  (g:triu self diagonal))

(define/contract-out (masked-fill self mask value)
  (-> tensor? tensor? real? tensor?)
  (g:masked-fill-scalar self mask (exact->inexact value)))

(define/contract-out (embedding indices weight #:padding-idx [padding-idx #f])
  (->* [tensor? tensor?]
       [#:padding-idx (or/c #f exact-nonnegative-integer?)]
       tensor?)
  (g:embedding weight indices (or padding-idx -1) #f #f))

(define/contract-out (layer-norm input normalized-shape
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
