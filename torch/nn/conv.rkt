#lang racket/base

(require (only-in racket/contract/base ->* or/c)
         (only-in "../foreign.rkt" conv1d conv2d flatten max-pool2d)
         (only-in "../foreign/contracts.rkt"
                  nonneg-size-1d/c nonneg-size/c pos-size-1d/c pos-size/c)
         (only-in "../foreign/size.rkt" ->1d ->2d)
         (only-in "init.rkt" fan-in kaiming-uniform uniform-init)
         (only-in "module.rkt" define-module))

(define-module Conv1d (in-channels out-channels kernel-size ;; noqa
                       #:stride [stride 1]
                       #:padding [padding 0]
                       #:dilation [dilation 1])
  #:contract (->* [exact-positive-integer? exact-positive-integer? pos-size-1d/c]
                  [#:stride pos-size-1d/c
                   #:padding nonneg-size-1d/c
                   #:dilation pos-size-1d/c]
                  conv1d?)
  #:coerce ([kernel-size (->1d kernel-size)]
            [stride (->1d stride)]
            [padding (->1d padding)]
            [dilation (->1d dilation)])
  ;; weight before bias: nn.Conv1d.reset_parameters' RNG draw order
  #:params ([weight (kaiming-uniform (list out-channels in-channels
                                           (car kernel-size)))]
            [bias (let ([bound (/ 1.0 (sqrt (fan-in (list out-channels
                                                          in-channels
                                                          (car kernel-size)))))])
                    (uniform-init (list out-channels) (- bound) bound))])
  #:forward (x)
  (conv1d x weight #:bias bias #:stride stride #:padding padding
          #:dilation dilation))

(define-module Conv2d (in-channels out-channels kernel-size ;; noqa
                       #:stride [stride 1]
                       #:padding [padding 0])
  #:contract (->* [exact-positive-integer? exact-positive-integer? pos-size/c]
                  [#:stride pos-size/c #:padding nonneg-size/c]
                  conv2d?)
  #:coerce ([kernel-size (->2d kernel-size)]
            [stride (->2d stride)]
            [padding (->2d padding)])
  ;; weight before bias: nn.Conv2d.reset_parameters' RNG draw order
  #:params ([weight (kaiming-uniform (list out-channels in-channels
                                            (car kernel-size) (cadr kernel-size)))]
            [bias (let ([bound (/ 1.0 (sqrt (fan-in (list out-channels in-channels
                                                          (car kernel-size)
                                                          (cadr kernel-size)))))])
                    (uniform-init (list out-channels) (- bound) bound))])
  #:forward (x)
  (conv2d x weight #:bias bias #:stride stride #:padding padding))

(define-module MaxPool2d (kernel-size ;; noqa
                          #:stride [stride #f]
                          #:padding [padding 0])
  #:contract (->* [pos-size/c]
                  [#:stride (or/c #f pos-size/c) #:padding nonneg-size/c]
                  max-pool2d?)
  #:forward (x)
  (max-pool2d x kernel-size #:stride stride #:padding padding))

(define-module Flatten (#:start-dim [start-dim 1] #:end-dim [end-dim -1]) ;; noqa
  #:contract (->* [] [#:start-dim exact-integer? #:end-dim exact-integer?]
                  flatten?)
  #:forward (x)
  (flatten x start-dim end-dim))
