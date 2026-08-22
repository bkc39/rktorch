#lang racket/base

(require (only-in ffi/unsafe
                  _bytes
                  _double
                  _enum
                  _fun
                  _int
                  _int64
                  _ptr
                  _uint64)
         (only-in ffi/vector _f32vector _f64vector _s64vector)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide dtype-code->symbol
         tr-tensor-copy-data-f64/raw
         tr-tensor-narrow/raw
         tr-tensor-numel/raw
         tr-tensor-ndim/raw
         tr-tensor-shape/raw
         tr-tensor-copy-data-i64/raw
         tr-tensor-copy-data/raw
         tr-tensor-dtype/raw
         tr-tensor-print/raw
         tr-tensor-item/raw
         tr-tensor-to-dtype/raw
         _tr-dtype)

(define-torch tr-tensor-numel/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_numel)

(define-torch tr-tensor-ndim/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_ndim)

(define-torch tr-tensor-shape/raw
  (_fun (t : _Tensor)
        (capacity : _int64)
        (out-dims : (_s64vector i))
        (out-ndim : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out-ndim))
  #:c-id tr_tensor_shape)

(define-torch tr-tensor-copy-data/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (out : (_f32vector i))
        (out-numel : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-numel))
  #:c-id tr_tensor_copy_data)

(define _tr-dtype
  (_enum '(float32 = 0 float64 = 1 int64 = 2 bool = 3)))

(define (dtype-code->symbol n)
  (case n
    [(0) 'float32]
    [(1) 'float64]
    [(2) 'int64]
    [(3) 'bool]
    [else #f]))

;; out is a plain _int, not _tr-dtype: on the error path the C side never
;; writes it, and an enum unmarshal of that garbage raises BEFORE the
;; caller can check rc.
(define-torch tr-tensor-dtype/raw
  (_fun (t : _Tensor) (out : (_ptr o _int)) -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_dtype)

(define-torch tr-tensor-copy-data-f64/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (out : (_f64vector i))
        (out-numel : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-numel))
  #:c-id tr_tensor_copy_data_f64)

(define-torch tr-tensor-copy-data-i64/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (out : (_s64vector i))
        (out-numel : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-numel))
  #:c-id tr_tensor_copy_data_i64)

(define-torch tr-tensor-narrow/raw
  (_fun (t : _Tensor)
        (dim : _int64)
        (start : _int64)
        (len : _int64)
        -> _Tensor/null)
  #:c-id tr_gen_narrow
  #:wrap tensor-allocator)

(define-torch tr-tensor-item/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _double))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_item)

(define-torch tr-tensor-to-dtype/raw
  (_fun (t : _Tensor) (dtype : _tr-dtype) -> _Tensor/null)
  #:c-id tr_tensor_to_dtype
  #:wrap tensor-allocator)

(define-torch tr-tensor-print/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id tr_tensor_print)
