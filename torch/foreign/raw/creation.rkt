#lang racket/base

;; Raw tensor constructors (creation.h). Every function returns a freshly
;; allocated _Tensor (or NULL on error); `tensor-allocator`
;; registers the GC finalizer, exactly like tr-randn/raw.

(require (only-in ffi/unsafe _double _fun _int64 _uint64)
         (only-in ffi/vector _f32vector _s64vector)
         (only-in "syntax.rkt" _Tensor/null define-torch tensor-allocator))

(provide tr-zeros/raw
         tr-ones/raw
         tr-full/raw
         tr-arange/raw
         tr-eye/raw
         tr-from-data/raw)

(define-torch tr-zeros/raw
  (_fun (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_zeros
  #:wrap tensor-allocator)

(define-torch tr-ones/raw
  (_fun (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_ones
  #:wrap tensor-allocator)

(define-torch tr-full/raw
  (_fun (dims : (_s64vector i)) (ndim : _int64) (value : _double) -> _Tensor/null)
  #:c-id tr_full
  #:wrap tensor-allocator)

(define-torch tr-arange/raw
  (_fun (start : _double) (end : _double) (step : _double) -> _Tensor/null)
  #:c-id tr_arange
  #:wrap tensor-allocator)

(define-torch tr-eye/raw
  (_fun (n : _int64) (m : _int64) -> _Tensor/null)
  #:c-id tr_eye
  #:wrap tensor-allocator)

(define-torch tr-from-data/raw
  (_fun (data : (_f32vector i))
        (numel : _uint64)
        (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_from_data
  #:wrap tensor-allocator)
