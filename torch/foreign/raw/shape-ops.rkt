#lang racket/base

;; Raw shape-manipulation ops (shape_ops.h). All return new GC-managed
;; _Tensor handles (views share storage at the ATen level, but each handle
;; owns its own reference, so finalization order doesn't matter).

(require (only-in ffi/unsafe _fun _int64 _list)
         (only-in ffi/unsafe/alloc allocator)
         (only-in ffi/vector _s64vector)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch tr-tensor-free/finalizer))

(provide tr-reshape/raw
         tr-view/raw
         tr-transpose/raw
         tr-permute/raw
         tr-squeeze/raw
         tr-squeeze-dim/raw
         tr-unsqueeze/raw
         tr-cat/raw
         tr-stack/raw)

(define-torch tr-reshape/raw
  (_fun (t : _Tensor) (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_reshape
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-view/raw
  (_fun (t : _Tensor) (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_view
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-transpose/raw
  (_fun (t : _Tensor) (dim0 : _int64) (dim1 : _int64) -> _Tensor/null)
  #:c-id tr_transpose
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-permute/raw
  (_fun (t : _Tensor) (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_permute
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-squeeze/raw
  (_fun (t : _Tensor) -> _Tensor/null)
  #:c-id tr_squeeze
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-squeeze-dim/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_squeeze_dim
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-unsqueeze/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_unsqueeze
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-cat/raw
  (_fun (ts : (_list i _Tensor)) (n : _int64) (dim : _int64) -> _Tensor/null)
  #:c-id tr_cat
  #:wrap (allocator tr-tensor-free/finalizer))

(define-torch tr-stack/raw
  (_fun (ts : (_list i _Tensor)) (n : _int64) (dim : _int64) -> _Tensor/null)
  #:c-id tr_stack
  #:wrap (allocator tr-tensor-free/finalizer))
