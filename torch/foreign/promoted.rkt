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

(require (only-in racket/list drop [flatten list-flatten] take)
         (only-in "ops.rkt" tensor-shape)
         (only-in "size.rkt" ->2d)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" reshape)
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
         ;; narrow needs no wrapper (no keyword defaulting, no dispatch); the
         ;; generated binding already has the right name and contract target.
         ;; Like torch.narrow it returns a *view* aliasing the source storage
         ;; (in-place writes to the result mutate the original); ATen
         ;; refcounting keeps storage alive regardless of GC order.
         (rename-out [g:narrow narrow]))

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
       [(zero? n)
        ;; A 0-d tensor admits only the trivial dim 0 (or -1, which PyTorch
        ;; normalizes to 0); anything else is out of range (PyTorch raises
        ;; IndexError) rather than a silent flatten to [1].
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
        ;; apply * over for/list, not for/product: the latter is a recent
        ;; racket/base addition, and this keeps no version floor.
        (define collapsed
          (apply * (for/list ([d (in-list shp)] [i (in-naturals)]
                                                 #:when (and (>= i s) (<= i e)))
                     d)))
        (apply reshape v (append (take shp s)
                                 (list collapsed)
                                 (drop shp (add1 e))))])]
    [else (list-flatten v)]))

;; --------------------------------------------------- comparison dispatchers

;; eq/ne/lt/le/gt/ge over a tensor lhs and a tensor-or-real rhs. The result
;; handle is a genuine bool tensor (what masked-fill demands); only the
;; read path (tensor->list / repr) coerces the values to float32.
(define ((comparison t-op s-op) a b)
  (cond
    [(tensor? b) (t-op a b)]
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

;; ------------------------------------------- transformer primitives (#22)

;; tril/triu with PyTorch's default diagonal. The GPT causal mask is
;; (eq (tril (ones T T)) 0) -- a bool tensor marking the *upper* triangle.
(define (tril self [diagonal 0])
  (g:tril self diagonal))

(define (triu self [diagonal 0])
  (g:triu self diagonal))

;; masked-fill: mask must be a bool tensor (a comparison-op result) --
;; ATen rejects float masks. `value` may be -inf.0 (the softmax mask).
(define (masked-fill self mask value)
  (g:masked-fill-scalar self mask (exact->inexact value)))

;; F.embedding argument order (indices first); the unstable surface keeps
;; ATen's weight-first order. #:padding-idx #f means "none" (ATen's -1).
(define (embedding input weight #:padding-idx [padding-idx #f])
  (g:embedding weight input (or padding-idx -1) #f #f))

;; F.layer_norm defaults: no affine params unless given, eps 1e-5;
;; cudnn_enable stays #t like torch.nn.functional.layer_norm.
;; normalized-shape takes an int (the trailing dim) or an explicit list.
(define (layer-norm input normalized-shape
                    #:weight [weight #f]
                    #:bias [bias #f]
                    #:eps [eps 1e-5])
  (define shape
    (if (list? normalized-shape) normalized-shape (list normalized-shape)))
  (g:layer-norm input shape weight bias eps #t))
