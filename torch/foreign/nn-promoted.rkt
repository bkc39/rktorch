#lang racket/base

(require (only-in "size.rkt" ->2d)
         (prefix-in g: (only-in "../generated.rkt"
                                adaptive-avg-pool2d
                                avg-pool2d
                                conv2d
                                embedding
                                layer-norm
                                masked-fill-scalar
                                max-pool2d
                                tril
                                triu)))

(provide adaptive-avg-pool2d avg-pool2d conv2d embedding layer-norm
         masked-fill max-pool2d tril triu)

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
