#lang racket/base

;; Hand-curated promotions of the generated surface (torch/generated.rkt,
;; uncontracted) into the contracted public facade: conv/pool wrappers
;; with PyTorch-style keyword defaults, comparison dispatchers over a
;; tensor-or-real rhs, and a shadow-dispatching `flatten`.  Contracts
;; live in ../foreign.rkt.

(require (only-in racket/list drop [flatten list-flatten] take)
         (only-in "ops.rkt" tensor-device tensor-dtype tensor-shape)
         (only-in "size.rkt" ->2d)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" reshape tensor)
         (prefix-in g: (only-in "../generated.rkt"
                                adaptive-avg-pool2d
                                avg-pool2d
                                conv2d
                                embedding
                                eq-scalar eq-tensor
                                ge-scalar ge-tensor
                                gt-scalar gt-tensor
                                layer-norm
                                le-scalar le-tensor
                                lt-scalar lt-tensor
                                masked-fill-scalar
                                max-pool2d
                                narrow
                                ne-scalar ne-tensor
                                tril
                                triu)))

(provide flatten
         eq ne lt le gt ge
         conv2d max-pool2d avg-pool2d adaptive-avg-pool2d
         tril triu masked-fill embedding layer-norm
         ;; narrow needs no wrapper. Like torch.narrow it returns a *view*
         ;; aliasing the source storage (in-place writes to the result
         ;; mutate the original); ATen refcounting keeps the storage alive
         ;; regardless of GC order.
         (rename-out [g:narrow narrow]))

;; ------------------------------------------------------------ flatten shim

;; Collapses dims start..end into one via reshape over the cached shape;
;; a non-tensor defers to racket/list's flatten.
(define (flatten v [start-dim 0] [end-dim -1])
  (cond
    [(tensor? v)
     (define shp (tensor-shape v))
     (define n (length shp))
     (cond
       [(zero? n)
        ;; A 0-d tensor admits only dim 0 (or -1, which PyTorch normalizes
        ;; to 0); anything else raises, as PyTorch's IndexError does.
        (unless (and (memv start-dim '(0 -1)) (memv end-dim '(0 -1)))
          (error 'flatten
                 "invalid dim range [~a, ~a] for a 0-d tensor"
                 start-dim end-dim))
        (reshape v 1)]
       [else
        (define s (if (negative? start-dim) (+ n start-dim) start-dim))
        (define e (if (negative? end-dim) (+ n end-dim) end-dim))
        (unless (and (<= 0 s e) (< e n))
          (error 'flatten
                 "invalid dim range [~a, ~a] for a ~a-d tensor"
                 start-dim end-dim n))
        ;; not for/product: a recent racket/base addition; this keeps no
        ;; version floor
        (define collapsed
          (apply * (for/list ([d (in-list shp)] [i (in-naturals)]
                                                 #:when (<= s i e))
                     d)))
        (apply reshape v (append (take shp s)
                                 (list collapsed)
                                 (drop shp (add1 e))))])]
    [else (list-flatten v)]))

;; --------------------------------------------------- comparison dispatchers

;; The result handle is a genuine bool tensor (what masked-fill demands);
;; only the read path coerces the values to float32.  An exact-integer rhs
;; against an int64 lhs routes through a TENSOR rhs: the scalar path
;; transits a C double, which rounds past 2^53 and could flip the
;; comparison RESULT.
(define ((comparison t-op s-op) a b)
  (cond
    [(tensor? b) (t-op a b)]
    [(and (exact-integer? b) (eq? (tensor-dtype a) 'int64))
     ;; construct the scalar BESIDE the lhs — the default device can
     ;; differ from a's, and a cross-device comparison errors
     (t-op a (tensor b #:device (tensor-device a)))]
    [else (s-op a (exact->inexact b))]))

(define eq (comparison g:eq-tensor g:eq-scalar))
(define ne (comparison g:ne-tensor g:ne-scalar))
(define lt (comparison g:lt-tensor g:lt-scalar))
(define le (comparison g:le-tensor g:le-scalar))
(define gt (comparison g:gt-tensor g:gt-scalar))
(define ge (comparison g:ge-tensor g:ge-scalar))

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

;; ------------------------------------------- transformer primitives

(define (tril self [diagonal 0])
  (g:tril self diagonal))

(define (triu self [diagonal 0])
  (g:triu self diagonal))

;; mask must be a bool tensor (a comparison-op result) — ATen rejects
;; float masks. `value` may be -inf.0.
(define (masked-fill self mask value)
  (g:masked-fill-scalar self mask (exact->inexact value)))

;; F.embedding argument order (indices first); the unstable surface keeps
;; ATen's weight-first order. #:padding-idx #f means "none" (ATen's -1).
(define (embedding input weight #:padding-idx [padding-idx #f])
  (g:embedding weight input (or padding-idx -1) #f #f))

;; F.layer_norm defaults: no affine params unless given, eps 1e-5,
;; cudnn_enable #t. normalized-shape takes an int or an explicit list.
(define (layer-norm input normalized-shape
                    #:weight [weight #f]
                    #:bias [bias #f]
                    #:eps [eps 1e-5])
  (define shape
    (if (list? normalized-shape) normalized-shape (list normalized-shape)))
  (g:layer-norm input shape weight bias eps #t))
