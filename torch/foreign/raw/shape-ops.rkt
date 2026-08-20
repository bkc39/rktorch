#lang racket/base

;; Raw shape-manipulation ops (shape_ops.h). All return new GC-managed
;; _Tensor handles (views share storage at the ATen level, but each handle
;; owns its own reference, so finalization order doesn't matter).

(require (only-in ffi/unsafe _fun _int64 _list)
         (only-in ffi/vector _s64vector)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

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
  #:wrap tensor-allocator)

(define-torch tr-view/raw
  (_fun (t : _Tensor) (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_view
  #:wrap tensor-allocator)

(define-torch tr-transpose/raw
  (_fun (t : _Tensor) (dim0 : _int64) (dim1 : _int64) -> _Tensor/null)
  #:c-id tr_transpose
  #:wrap tensor-allocator)

(define-torch tr-permute/raw
  (_fun (t : _Tensor) (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
  #:c-id tr_permute
  #:wrap tensor-allocator)

(define-torch tr-squeeze/raw
  (_fun (t : _Tensor) -> _Tensor/null)
  #:c-id tr_squeeze
  #:wrap tensor-allocator)

(define-torch tr-squeeze-dim/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_squeeze_dim
  #:wrap tensor-allocator)

(define-torch tr-unsqueeze/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_unsqueeze
  #:wrap tensor-allocator)

(define-torch tr-cat/raw
  (_fun (ts : (_list i _Tensor)) (n : _int64) (dim : _int64) -> _Tensor/null)
  #:c-id tr_cat
  #:wrap tensor-allocator)

(define-torch tr-stack/raw
  (_fun (ts : (_list i _Tensor)) (n : _int64) (dim : _int64) -> _Tensor/null)
  #:c-id tr_stack
  #:wrap tensor-allocator)
