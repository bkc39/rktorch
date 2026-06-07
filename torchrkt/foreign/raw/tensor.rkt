#lang racket/base

;; Tensor: opaque handle to a torch::Tensor.
;;
;; `define-cpointer-type` generates three names:
;;   _Tensor       — non-null cpointer type (tag 'Tensor)
;;   _Tensor/null  — nullable cpointer type (used for NULL-on-error returns)
;;   Tensor?       — predicate
;;
;; The deallocator must be defined before any allocator that references it
;; (tr-randn/raw in random.rkt wraps with `(allocator tr-tensor-free/raw)`).

(require ffi/unsafe
         ffi/unsafe/alloc
         ffi/vector
         "library.rkt")

(provide _Tensor
         _Tensor/null ;; noqa
         Tensor? ;; noqa
         tr-tensor-free/raw
         tr-tensor-numel/raw
         tr-tensor-ndim/raw
         tr-tensor-shape/raw
         tr-tensor-copy-data/raw
         tr-tensor-print/raw)

(define-cpointer-type _Tensor)

(define-torchrkt tr-tensor-free/raw
  (_fun _Tensor -> _void)
  #:c-id tr_tensor_free
  #:wrap (deallocator))

(define-torchrkt tr-tensor-numel/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_numel)

(define-torchrkt tr-tensor-ndim/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_tensor_ndim)

;; (raw t capacity out-dims) -> (values rc out-ndim).  out-dims is a caller
;; allocated s64vector written in place; out-ndim always holds the true ndim.
(define-torchrkt tr-tensor-shape/raw
  (_fun (t : _Tensor)
        (capacity : _int64)
        (out-dims : (_s64vector i))
        (out-ndim : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out-ndim))
  #:c-id tr_tensor_shape)

;; (raw t capacity out) -> (values rc out-numel).  out is a caller-allocated
;; f32vector written in place; out-numel always holds the true element count.
(define-torchrkt tr-tensor-copy-data/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (out : (_f32vector i))
        (out-numel : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-numel))
  #:c-id tr_tensor_copy_data)

;; (raw t capacity buf) -> (values rc out-len), size-then-fill string probe.
(define-torchrkt tr-tensor-print/raw
  (_fun (t : _Tensor)
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id tr_tensor_print)
