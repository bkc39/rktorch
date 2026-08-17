#lang racket/base

;; Raw tensor constructors. `tr-randn/raw` returns a freshly-allocated _Tensor
;; (or NULL on error); `(allocator tr-tensor-free/finalizer)` registers the GC
;; finalizer so the handle is reclaimed automatically, exactly like xgboost's
;; DMatrix/Booster constructors.

(require (only-in ffi/unsafe _double _fun _int _int64)
         (only-in ffi/unsafe/alloc allocator)
         (only-in ffi/vector _s64vector)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch tr-tensor-free/finalizer))

(provide tr-randn/raw
         tr-rand/raw
         tr-tensor-uniform!/raw)

(define-torch tr-randn/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_randn
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-rand/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_rand
  #:wrap (allocator tr-tensor-free/finalizer))

;; In-place fill with uniform draws on [low, high); consumes the global RNG
;; exactly like torch.Tensor.uniform_, which nn init parity depends on.
(define-torch tr-tensor-uniform!/raw
  (_fun (t : _Tensor) (low : _double) (high : _double) -> _int)
  #:c-id tr_tensor_uniform_)
