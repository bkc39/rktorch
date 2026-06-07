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

(provide tr-randn/raw)

(define-torchrkt tr-randn/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_randn
  #:wrap (allocator tr-tensor-free/raw))
