#lang racket/base

;; Safe tensor operations built on the raw layer + wrapper struct.  These are
;; the implementation behind the contracts in ../foreign.rkt.

(require (only-in ffi/vector f32vector->list list->s64vector make-f32vector)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/global.rkt" tr-last-error/raw tr-manual-seed/raw tr-version/raw)
         (only-in "raw/random.rkt" tr-rand/raw tr-randn/raw tr-tensor-uniform!/raw)
         (only-in "raw/tensor.rkt"
                  tr-tensor-copy-data/raw
                  tr-tensor-item/raw
                  tr-tensor-numel/raw
                  tr-tensor-to-dtype/raw)
         (only-in "structs.rkt"
                  handle->repr
                  handle->string
                  tensor-handle
                  tensor-impl-shape
                  wrap-tensor))

(provide torch-version
         manual-seed!
         randn
         rand
         uniform!
         item
         to-dtype
         tensor-numel
         tensor-shape
         tensor->vector
         tensor->list
         tensor->string
         tensor->repr)

(define (torch-version)
  (tr-version/raw))

(define (manual-seed! seed)
  (check-ok (tr-manual-seed/raw seed) 'manual-seed!)
  (void))

(define (randn . dims)
  (define h (tr-randn/raw (list->s64vector dims) (length dims)))
  (unless h
    (error 'randn "randn failed: ~a" (tr-last-error/raw)))
  (wrap-tensor h))

;; Uniform draws on [0, 1), torch.rand.
(define (rand . dims)
  (wrap-tensor
   (check-handle 'rand (tr-rand/raw (list->s64vector dims) (length dims)))))

;; In-place fill with uniform draws on [low, high) — torch.Tensor.uniform_,
;; the RNG primitive behind PyTorch's nn.Linear init.
(define (uniform! t low high)
  (check-ok (tr-tensor-uniform!/raw t
                                    (exact->inexact low)
                                    (exact->inexact high))
            'uniform!)
  (void))

;; The value of a one-element tensor as a Racket real (torch.Tensor.item).
(define (item t)
  (define-values (rc v) (tr-tensor-item/raw t))
  (check-ok rc 'item)
  v)

;; Copy converted to 'float32 / 'float64 / 'int64 (torch.Tensor.to).
(define (to-dtype t dtype)
  (wrap-tensor (check-handle 'to-dtype (tr-tensor-to-dtype/raw t dtype))))

(define (tensor-numel t)
  (define-values (rc n) (tr-tensor-numel/raw t))
  (check-ok rc 'tensor-numel)
  n)

;; Cached at wrap time, so no C round-trip.
(define (tensor-shape t)
  (tensor-impl-shape t))

(define (tensor->vector t)
  (define numel (tensor-numel t))
  (define out (make-f32vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data/raw t numel out))
  (check-ok rc 'tensor->vector)
  out)

(define (tensor->list t)
  (f32vector->list (tensor->vector t)))

;; ATen's C++ `operator<<` text (the libtorch-native printer).
(define (tensor->string t)
  (handle->string (tensor-handle t)))

;; The PyTorch `repr` text -- identical to what the REPL prints (see structs.rkt).
(define (tensor->repr t)
  (handle->repr (tensor-handle t) (tensor-shape t)))
