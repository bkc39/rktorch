#lang racket/base

;; Raw tensor constructors. `tr-randn/raw` returns a freshly-allocated _Tensor
;; (or NULL on error); `(allocator tr-tensor-free/raw)` registers the GC
;; finalizer so the handle is reclaimed automatically, exactly like xgboost's
;; DMatrix/Booster constructors.

(require ffi/unsafe
         ffi/unsafe/alloc
         ffi/vector
         "library.rkt"
         "tensor.rkt")

(provide tr-randn/raw
         tr-rand/raw
         tr-tensor-uniform!/raw)

(define-torchrkt tr-randn/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_randn
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-rand/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_rand
  #:wrap (allocator tr-tensor-free/raw))

;; In-place fill with uniform draws on [low, high); consumes the global RNG
;; exactly like torch.Tensor.uniform_, which nn init parity depends on.
(define-torchrkt tr-tensor-uniform!/raw
  (_fun (t : _Tensor) (low : _double) (high : _double) -> _int)
  #:c-id tr_tensor_uniform_)
