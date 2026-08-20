#lang racket/base

;; Raw reductions (reduce.h). Whole-tensor forms return 0-dim tensors; argmax
;; returns int64 tensors (read them via tr-tensor-item/raw or copy-data's
;; float32 conversion).

(require (only-in ffi/unsafe _bool _fun _int64)
         (only-in "memory.rkt" define-unary/raw tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-sum/raw
         tr-mean/raw
         tr-max/raw
         tr-min/raw
         tr-argmax-all/raw
         tr-argmax/raw
         tr-softmax/raw
         tr-log-softmax/raw)

(define-unary/raw tr-sum/raw tr_sum)
(define-unary/raw tr-mean/raw tr_mean)
(define-unary/raw tr-max/raw tr_max)
(define-unary/raw tr-min/raw tr_min)
(define-unary/raw tr-argmax-all/raw tr_argmax_all)

(define-torch tr-argmax/raw
  (_fun (t : _Tensor) (dim : _int64) (keepdim : _bool) -> _Tensor/null)
  #:c-id tr_argmax
  #:wrap tensor-allocator)

(define-torch tr-softmax/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_softmax
  #:wrap tensor-allocator)

(define-torch tr-log-softmax/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_log_softmax
  #:wrap tensor-allocator)
