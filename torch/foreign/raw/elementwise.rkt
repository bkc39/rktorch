#lang racket/base

;; Raw elementwise ops (elementwise.h), via the shared op-definer macros
;; in syntax.rkt (binary tensor-tensor, tensor-scalar, unary).

(require (only-in "syntax.rkt"
                  define-binary/raw
                  define-scalar/raw
                  define-unary/raw))

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
         tr-tanh/raw
         tr-gelu/raw)

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
(define-unary/raw tr-gelu/raw tr_gelu)
