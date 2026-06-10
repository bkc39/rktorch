#lang racket/base

;; Raw elementwise ops (elementwise.h). Three uniform signatures — binary
;; tensor-tensor, tensor-scalar, and unary — so a local definer macro per
;; shape replaces the hand-copied _fun boilerplate.

(require ffi/unsafe
         ffi/unsafe/alloc
         "library.rkt"
         "tensor.rkt")

(provide tr-add/raw
         tr-sub/raw
         tr-mul/raw
         tr-div/raw
         tr-pow/raw
         tr-add-scalar/raw
         tr-sub-scalar/raw
         tr-mul-scalar/raw
         tr-div-scalar/raw
         tr-pow-scalar/raw
         tr-neg/raw
         tr-exp/raw
         tr-log/raw
         tr-sqrt/raw
         tr-relu/raw
         tr-sigmoid/raw
         tr-tanh/raw)

(define-syntax-rule (define-binary/raw name c-id)
  (define-torchrkt name
    (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
    #:c-id c-id
    #:wrap (allocator tr-tensor-free/raw)))

(define-syntax-rule (define-scalar/raw name c-id)
  (define-torchrkt name
    (_fun (a : _Tensor) (b : _double) -> _Tensor/null)
    #:c-id c-id
    #:wrap (allocator tr-tensor-free/raw)))

(define-syntax-rule (define-unary/raw name c-id)
  (define-torchrkt name
    (_fun (t : _Tensor) -> _Tensor/null)
    #:c-id c-id
    #:wrap (allocator tr-tensor-free/raw)))

(define-binary/raw tr-add/raw tr_add)
(define-binary/raw tr-sub/raw tr_sub)
(define-binary/raw tr-mul/raw tr_mul)
(define-binary/raw tr-div/raw tr_div)
(define-binary/raw tr-pow/raw tr_pow)

(define-scalar/raw tr-add-scalar/raw tr_add_scalar)
(define-scalar/raw tr-sub-scalar/raw tr_sub_scalar)
(define-scalar/raw tr-mul-scalar/raw tr_mul_scalar)
(define-scalar/raw tr-div-scalar/raw tr_div_scalar)
(define-scalar/raw tr-pow-scalar/raw tr_pow_scalar)

(define-unary/raw tr-neg/raw tr_neg)
(define-unary/raw tr-exp/raw tr_exp)
(define-unary/raw tr-log/raw tr_log)
(define-unary/raw tr-sqrt/raw tr_sqrt)
(define-unary/raw tr-relu/raw tr_relu)
(define-unary/raw tr-sigmoid/raw tr_sigmoid)
(define-unary/raw tr-tanh/raw tr_tanh)
