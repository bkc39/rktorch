#lang racket/base

;; Hand-curated promotions of the generated surface into the contracted
;; public facade. The codegen generator emits uncontracted bindings into
;; torch/generated.rkt (the unstable surface); this module wraps the ones
;; whose ergonomics are settled (issue #3): ergonomic conv/pool wrappers
;; with PyTorch-style keyword defaults, comparison dispatchers that take a
;; tensor or real rhs, and a `flatten` that shadow-dispatches like
;; max/min/argmax (tensors collapse dims; anything else defers to
;; racket/list). The Adam in-place family and loss primitives stay
;; uncontracted in torch/generated.rkt -- the nn layer (#4) wraps those.
;;
;; Contracts live in ../foreign.rkt.

(require (only-in racket/list [flatten list-flatten] take drop)
         (only-in "ops.rkt" tensor-shape)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" reshape)
         (prefix-in g: (only-in "../generated.rkt"
                                adaptive-avg-pool2d
                                avg-pool2d
                                conv2d
                                eq-scalar eq-tensor
                                ge-scalar ge-tensor
                                gt-scalar gt-tensor
                                le-scalar le-tensor
                                lt-scalar lt-tensor
                                max-pool2d
                                narrow
                                ne-scalar ne-tensor)))

(provide flatten
         eq ne lt le gt ge
         conv2d max-pool2d avg-pool2d adaptive-avg-pool2d narrow)

;; A pooling/conv size arg is an int (broadcast to a square) or an explicit
;; [h w] list, mirroring PyTorch.
(define (->2d x) (if (list? x) x (list x x)))

;; ------------------------------------------------------------ flatten shim

;; (flatten tensor [start-dim] [end-dim]) collapses dims start..end into one,
;; via reshape over the queried shape -- so no new C op is needed. Given a
;; non-tensor it defers to racket/list's flatten, keeping `(require torch)`
;; safe for list code (the max/min/argmax convention).
(define (flatten v [start-dim 0] [end-dim -1])
  (cond
    [(tensor? v)
     (define shp (tensor-shape v))
     (define n (length shp))
     (cond
       [(zero? n) (reshape v 1)]
       [else
        (define s (if (negative? start-dim) (+ n start-dim) start-dim))
        (define e (if (negative? end-dim) (+ n end-dim) end-dim))
        (define collapsed
          (for/product ([d (in-list shp)] [i (in-naturals)]
                                          #:when (and (>= i s) (<= i e)))
            d))
        (apply reshape v (append (take shp s)
                                 (list collapsed)
                                 (drop shp (add1 e))))])]
    [else (list-flatten v)]))

;; --------------------------------------------------- comparison dispatchers

;; eq/ne/lt/le/gt/ge over a tensor lhs and a tensor-or-real rhs. They yield
;; float32 masks today (the read path coerces the bool result); int/bool
;; dtype is v3 work.
(define ((comparison who t-op s-op) a b)
  (cond
    [(tensor? b) (t-op a b)]
    [else (s-op a (exact->inexact b))]))

(define eq (comparison 'eq g:eq-tensor g:eq-scalar))
(define ne (comparison 'ne g:ne-tensor g:ne-scalar))
(define lt (comparison 'lt g:lt-tensor g:lt-scalar))
(define le (comparison 'le g:le-tensor g:le-scalar))
(define gt (comparison 'gt g:gt-tensor g:gt-scalar))
(define ge (comparison 'ge g:ge-tensor g:ge-scalar))

;; --------------------------------------------------- conv + pooling wrappers

;; PyTorch-style defaults; stride/padding/dilation accept an int or [h w].
(define (conv2d input weight
                #:bias [bias #f]
                #:stride [stride 1]
                #:padding [padding 0]
                #:dilation [dilation 1]
                #:groups [groups 1])
  (g:conv2d input weight bias
            (->2d stride) (->2d padding) (->2d dilation) groups))

;; stride defaults to kernel-size, like nn.MaxPool2d.
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

(define (narrow input dim start length)
  (g:narrow input dim start length))
