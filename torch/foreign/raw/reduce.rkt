#lang racket/base

;; Raw reductions (reduce.h). Whole-tensor forms return 0-dim tensors; argmax
;; returns int64 tensors (read them via tr-tensor-item/raw or copy-data's
;; float32 conversion).

(require (only-in ffi/unsafe _bool _fun _int64)
         (only-in ffi/unsafe/alloc allocator)
         (only-in "syntax.rkt" define-torchrkt)
         (only-in "tensor.rkt" _Tensor _Tensor/null tr-tensor-free/raw))

(provide tr-sum/raw
         tr-mean/raw
         tr-max/raw
         tr-min/raw
         tr-argmax-all/raw
         tr-argmax/raw
         tr-softmax/raw
         tr-log-softmax/raw)

(define-syntax-rule (define-whole/raw name c-id)
  (define-torchrkt name
    (_fun (t : _Tensor) -> _Tensor/null)
    #:c-id c-id
    #:wrap (allocator tr-tensor-free/raw)))

(define-whole/raw tr-sum/raw tr_sum)
(define-whole/raw tr-mean/raw tr_mean)
(define-whole/raw tr-max/raw tr_max)
(define-whole/raw tr-min/raw tr_min)
(define-whole/raw tr-argmax-all/raw tr_argmax_all)

(define-torchrkt tr-argmax/raw
  (_fun (t : _Tensor) (dim : _int64) (keepdim : _bool) -> _Tensor/null)
  #:c-id tr_argmax
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-softmax/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_softmax
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-log-softmax/raw
  (_fun (t : _Tensor) (dim : _int64) -> _Tensor/null)
  #:c-id tr_log_softmax
  #:wrap (allocator tr-tensor-free/raw))
