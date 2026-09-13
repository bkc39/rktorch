#lang racket/base

(require (only-in racket/contract/base ->* ->i or/c unsupplied-arg?)
         (only-in "../foreign.rkt"
                  conv-transpose2d conv1d conv2d flatten max-pool2d)
         (only-in "../foreign/contracts.rkt"
                  nonneg-size-1d/c nonneg-size/c pos-size-1d/c pos-size/c)
         (only-in "../foreign/size.rkt" ->1d ->2d)
         (only-in "init.rkt" fan-in kaiming-uniform uniform-init)
         (only-in "layer.rkt" define-layer)
         (only-in "parameter.rkt" Parameter))

(define-layer Conv1d (kernel-size stride padding dilation weight bias) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer? pos-size-1d/c]
                  [#:stride pos-size-1d/c
                   #:padding nonneg-size-1d/c
                   #:dilation pos-size-1d/c]
                  conv1d?)
  #:init (in-channels out-channels kernel-size
          #:stride [stride 1]
          #:padding [padding 0]
          #:dilation [dilation 1])
  (set! kernel-size (->1d kernel-size))
  (set! stride (->1d stride))
  (set! padding (->1d padding))
  (set! dilation (->1d dilation))
  (define shape (list out-channels in-channels (car kernel-size)))
  ;; weight before bias: nn.Conv1d.reset_parameters' RNG draw order
  (set! weight (Parameter (kaiming-uniform shape)))
  (set! bias
        (let ([bound (/ 1.0 (sqrt (fan-in shape)))])
          (Parameter (uniform-init (list out-channels) (- bound) bound))))
  #:forward (x)
  (conv1d x weight #:bias bias #:stride stride #:padding padding
          #:dilation dilation))

(define-layer Conv2d (kernel-size stride padding weight bias) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer? pos-size/c]
                  [#:stride pos-size/c #:padding nonneg-size/c]
                  conv2d?)
  #:init (in-channels out-channels kernel-size
          #:stride [stride 1]
          #:padding [padding 0])
  (set! kernel-size (->2d kernel-size))
  (set! stride (->2d stride))
  (set! padding (->2d padding))
  (define shape ;; noqa
    (list out-channels in-channels (car kernel-size) (cadr kernel-size)))
  ;; weight before bias: nn.Conv2d.reset_parameters' RNG draw order
  (set! weight (Parameter (kaiming-uniform shape)))
  (set! bias
        (let ([bound (/ 1.0 (sqrt (fan-in shape)))])
          (Parameter (uniform-init (list out-channels) (- bound) bound))))
  #:forward (x)
  (conv2d x weight #:bias bias #:stride stride #:padding padding))

(define-layer ConvTranspose2d ;; noqa
  (kernel-size stride padding output-padding dilation groups weight bias)
  #:contract (->i ([in-channels exact-positive-integer?]
                   [out-channels exact-positive-integer?]
                   [kernel-size pos-size/c])
                  (#:stride [stride pos-size/c]
                   #:padding [padding nonneg-size/c]
                   #:output-padding [output-padding nonneg-size/c]
                   #:dilation [dilation pos-size/c]
                   #:groups [groups exact-positive-integer?])
                  #:pre/name (in-channels out-channels groups)
                  "groups must divide both channel counts"
                  (or (unsupplied-arg? groups)
                      (and (zero? (remainder in-channels groups))
                           (zero? (remainder out-channels groups))))
                  [result conv-transpose2d?])
  #:init (in-channels out-channels kernel-size
          #:stride [stride 1]
          #:padding [padding 0]
          #:output-padding [output-padding 0]
          #:dilation [dilation 1]
          #:groups [groups 1])
  (set! kernel-size (->2d kernel-size))
  (set! stride (->2d stride))
  (set! padding (->2d padding))
  (set! output-padding (->2d output-padding))
  (set! dilation (->2d dilation))
  ;; the transposed layout (in, out/groups, kH, kW): fan-in reads out/groups
  (define shape ;; noqa
    (list in-channels (quotient out-channels groups)
          (car kernel-size) (cadr kernel-size)))
  ;; weight before bias: _ConvNd.reset_parameters' RNG draw order
  (set! weight (Parameter (kaiming-uniform shape)))
  (set! bias
        (let ([bound (/ 1.0 (sqrt (fan-in shape)))])
          (Parameter (uniform-init (list out-channels) (- bound) bound))))
  #:forward (x)
  (conv-transpose2d x weight #:bias bias #:stride stride #:padding padding
                    #:output-padding output-padding #:dilation dilation
                    #:groups groups))

(define-layer MaxPool2d (kernel-size ;; noqa
                         #:stride [stride #f]
                         #:padding [padding 0])
  #:contract (->* [pos-size/c]
                  [#:stride (or/c #f pos-size/c) #:padding nonneg-size/c]
                  max-pool2d?)
  #:forward (x)
  (max-pool2d x kernel-size #:stride stride #:padding padding))

(define-layer Flatten (#:start-dim [start-dim 1] #:end-dim [end-dim -1]) ;; noqa
  #:contract (->* [] [#:start-dim exact-integer? #:end-dim exact-integer?]
                  flatten?)
  #:forward (x)
  (flatten x start-dim end-dim))
