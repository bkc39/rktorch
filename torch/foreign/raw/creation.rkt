#lang racket/base

;; Raw tensor constructors (creation.h).

(require (only-in ffi/unsafe _double _fun _int64 _uint64)
         (only-in ffi/vector _f32vector _s64vector)
         (only-in "memory.rkt" _tr-device-type tensor-allocator)
         (only-in "syntax.rkt" _Tensor/null define-torch))

(provide tr-zeros/raw
         tr-ones/raw
         tr-full/raw
         tr-arange/raw
         tr-eye/raw
         tr-from-data/raw
         tr-from-data-i64-on/raw
         tr-from-data-i64/raw
         tr-from-data-on/raw)

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

;; EXPLICIT device: placement never routes host data through the process
;; default (a CUDA default would cost an explicitly-CPU tensor a
;; host->GPU->CPU bounce, or a CUDA OOM).
(define-torch tr-from-data-on/raw
  (_fun (data : (_f32vector i))
        (numel : _uint64)
        (dims : (_s64vector i))
        (ndim : _int64)
        (device-type : _tr-device-type)
        (device-index : _int64)
        -> _Tensor/null)
  #:c-id tr_from_data_on
  #:wrap tensor-allocator)

;; int64 ingestion pair (#44): int64 payload and dtype, so exact integers
;; never transit float32.
(define-torch tr-from-data-i64/raw
  (_fun (data : (_s64vector i))
        (numel : _uint64)
        (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_from_data_i64
  #:wrap tensor-allocator)

(define-torch tr-from-data-i64-on/raw
  (_fun (data : (_s64vector i))
        (numel : _uint64)
        (dims : (_s64vector i))
        (ndim : _int64)
        (device-type : _tr-device-type)
        (device-index : _int64)
        -> _Tensor/null)
  #:c-id tr_from_data_i64_on
  #:wrap tensor-allocator)
