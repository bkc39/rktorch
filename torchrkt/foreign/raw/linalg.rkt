#lang racket/base

;; Raw linear-algebra ops (linalg.h).

(require ffi/unsafe
         ffi/unsafe/alloc
         "library.rkt"
         "tensor.rkt")

(provide tr-matmul/raw
         tr-mm/raw
         tr-mv/raw
         tr-dot/raw)

(define-syntax-rule (define-binary/raw name c-id)
  (define-torchrkt name
    (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
    #:c-id c-id
    #:wrap (allocator tr-tensor-free/raw)))

(define-binary/raw tr-matmul/raw tr_matmul)
(define-binary/raw tr-mm/raw tr_mm)
(define-binary/raw tr-mv/raw tr_mv)
(define-binary/raw tr-dot/raw tr_dot)
