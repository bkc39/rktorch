#lang racket/base

(require (only-in racket/contract/base ->* </c >=/c and/c)
         (only-in "../generated.rkt" dropout)
         (only-in "layer.rkt" define-layer training? with-mode))

(define-layer Dropout (p) ;; noqa
  #:contract (->* [] [#:p (and/c (>=/c 0) (</c 1))] dropout?)
  #:init (#:p [p 0.5])
  #:forward (x)
  (with-mode (dropout x p (training? mode))))
