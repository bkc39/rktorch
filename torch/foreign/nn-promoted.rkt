#lang racket/base

(require (only-in racket/contract ->* define/contract or/c)
         (only-in "contracts.rkt" nonneg-size-1d/c pos-size-1d/c)
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

(provide adaptive-avg-pool2d avg-pool2d conv1d conv2d embedding layer-norm
         masked-fill max-pool2d tril triu)

(define/contract (conv1d input weight
                         #:bias [bias #f]
                         #:stride [stride 1]
                         #:padding [padding 0]
                         #:dilation [dilation 1]
                         #:groups [groups 1])
  (->* (tensor? tensor?)
       (#:bias (or/c tensor? #f) #:stride pos-size-1d/c
        #:padding nonneg-size-1d/c #:dilation pos-size-1d/c
        #:groups exact-positive-integer?)
       tensor?)
  (g:conv1d input weight bias
            (->1d stride) (->1d padding) (->1d dilation) groups))

(define (conv2d input weight
                #:bias [bias #f]
                #:stride [stride 1]
                #:padding [padding 0]
                #:dilation [dilation 1]
                #:groups [groups 1])
  (g:conv2d input weight bias
            (->2d stride) (->2d padding) (->2d dilation) groups))

(define (max-pool2d input kernel-size
                    #:stride [stride #f]
                    #:padding [padding 0]
                    #:dilation [dilation 1]
                    #:ceil-mode [ceil-mode #f])
  (g:max-pool2d input
                (->2d kernel-size)
                (->2d (or stride kernel-size))
                (->2d padding) (->2d dilation) ceil-mode))

(define (avg-pool2d input kernel-size
                    #:stride [stride #f]
                    #:padding [padding 0]
                    #:ceil-mode [ceil-mode #f]
                    #:count-include-pad [count-include-pad #t]
                    #:divisor-override [divisor-override #f])
  (g:avg-pool2d input
                (->2d kernel-size)
                (->2d (or stride kernel-size))
                (->2d padding) ceil-mode count-include-pad divisor-override))

(define (adaptive-avg-pool2d input output-size)
  (g:adaptive-avg-pool2d input (->2d output-size)))

(define (tril self [diagonal 0])
  (g:tril self diagonal))

(define (triu self [diagonal 0])
  (g:triu self diagonal))

(define (masked-fill self mask value)
  (g:masked-fill-scalar self mask (exact->inexact value)))

(define (embedding indices weight #:padding-idx [padding-idx #f])
  (g:embedding weight indices (or padding-idx -1) #f #f))

(define (layer-norm input normalized-shape
                    #:weight [weight #f]
                    #:bias [bias #f]
                    #:eps [eps 1e-5])
  (define shape
    (if (list? normalized-shape) normalized-shape (list normalized-shape)))
  (g:layer-norm input shape weight bias eps #t))
