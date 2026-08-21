#lang racket/base

;; Raw accessors over the opaque _Tensor handle (the handle type itself
;; and its deallocator live in syntax.rkt with the rest of the FFI
;; substrate).

(require (only-in ffi/unsafe
                  _bytes
                  _double
                  _enum
                  _fun
                  _int
                  _int64
                  _ptr
                  _uint64)
         (only-in ffi/vector _f32vector _s64vector)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-tensor-numel/raw
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

;; (raw t capacity out-dims) -> (values rc out-ndim).  out-dims is a caller
;; allocated s64vector written in place; out-ndim always holds the true ndim.
(define-torch tr-tensor-shape/raw
  (_fun (t : _Tensor)
        (capacity : _int64)
        (out-dims : (_s64vector i))
        (out-ndim : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out-ndim))
  #:c-id tr_tensor_shape)

;; (raw t capacity out) -> (values rc out-numel).  out is a caller-allocated
;; f32vector written in place; out-numel always holds the true element count.
(define-torch tr-tensor-copy-data/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (out : (_f32vector i))
        (out-numel : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-numel))
  #:c-id tr_tensor_copy_data)

;; Mirrors the tr_dtype C enum (tensor.h).
(define _tr-dtype
  (_enum '(float32 = 0 float64 = 1 int64 = 2)))

;; The out param is a plain _int, NOT _tr-dtype: on the error path (a
;; dtype outside the enum, e.g. a bool comparison mask) the C side never
;; writes the out param, and an enum unmarshal of that garbage raises
;; BEFORE the caller can check rc. Callers convert after the rc check
;; (dtype-int->symbol in ops.rkt).
(define-torch tr-tensor-dtype/raw
  (_fun (t : _Tensor) (out : (_ptr o _int)) -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_dtype)

;; tr-tensor-copy-data/raw's int64 sibling (#44): copies via a CPU/int64
;; conversion so integer tensors round-trip exactly.
(define-torch tr-tensor-copy-data-i64/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (out : (_s64vector i))
        (out-numel : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-numel))
  #:c-id tr_tensor_copy_data_i64)

(define-torch tr-tensor-item/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _double))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_item)

;; to-dtype allocates a fresh handle, so it carries the allocator wrap even
;; though it lives with the accessors.
(define-torch tr-tensor-to-dtype/raw
  (_fun (t : _Tensor) (dtype : _tr-dtype) -> _Tensor/null)
  #:c-id tr_tensor_to_dtype
  #:wrap tensor-allocator)

;; (raw t capacity buf) -> (values rc out-len), size-then-fill string probe.
(define-torch tr-tensor-print/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id tr_tensor_print)
