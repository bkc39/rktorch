#lang racket/base

;; Raw tensor constructors (creation.h). Every function returns a freshly
;; allocated _Tensor (or NULL on error); `(allocator tr-tensor-free/raw)`
;; registers the GC finalizer, exactly like tr-randn/raw.

(require ffi/unsafe
         ffi/unsafe/alloc
         ffi/vector
         "library.rkt"
         "tensor.rkt")

(provide tr-zeros/raw
         tr-ones/raw
         tr-full/raw
         tr-arange/raw
         tr-eye/raw
         tr-from-data/raw)

(define-torchrkt tr-zeros/raw
  (_fun (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_zeros
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-ones/raw
  (_fun (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_ones
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-full/raw
  (_fun (dims : (_s64vector i)) (ndim : _int64) (value : _double) -> _Tensor/null)
  #:c-id tr_full
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-arange/raw
  (_fun (start : _double) (end : _double) (step : _double) -> _Tensor/null)
  #:c-id tr_arange
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-eye/raw
  (_fun (n : _int64) (m : _int64) -> _Tensor/null)
  #:c-id tr_eye
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-from-data/raw
  (_fun (data : (_f32vector i))
        (numel : _uint64)
        (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_from_data
  #:wrap (allocator tr-tensor-free/raw))
