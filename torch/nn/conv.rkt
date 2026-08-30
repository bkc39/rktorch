#lang racket/base

(require (only-in "../foreign.rkt" conv1d conv2d flatten max-pool2d)
         (only-in "../foreign/size.rkt" ->1d ->2d)
         (only-in "init.rkt" fan-in kaiming-uniform uniform-init)
         (only-in "module.rkt" define-module))

(provide Conv1d
         (rename-out [Conv1d? conv1d?]) ;; noqa
         Conv2d
         (rename-out [Conv2d? conv2d?]) ;; noqa
         MaxPool2d
         (rename-out [MaxPool2d? max-pool2d?]) ;; noqa
         Flatten
         (rename-out [Flatten? flatten?]) ;; noqa
         )

(define-module Conv1d (in-channels out-channels kernel-size
                       #:stride [stride 1]
                       #:padding [padding 0]
                       #:dilation [dilation 1])
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

(define-module Conv2d (in-channels out-channels kernel-size
                       #:stride [stride 1]
                       #:padding [padding 0])
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

(define-module MaxPool2d (kernel-size
                          #:stride [stride #f]
                          #:padding [padding 0])
  #:forward (x)
  (max-pool2d x kernel-size #:stride stride #:padding padding))

(define-module Flatten (#:start-dim [start-dim 1] #:end-dim [end-dim -1])
  #:forward (x)
  (flatten x start-dim end-dim))
