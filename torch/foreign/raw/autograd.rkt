#lang racket/base

;; Raw autograd bindings (autograd.h). tr-tensor-grad/raw returns a fresh
;; tr_tensor box SHARING storage with the live .grad — freeing the box
;; releases one reference, not the gradient itself.

(require (only-in ffi/unsafe _bool _double _fun _int _ptr)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-tensor-requires-grad!/raw
         tr-tensor-requires-grad/raw
         tr-tensor-has-grad/raw
         tr-tensor-backward/raw
         tr-tensor-grad/raw
         tr-tensor-detach/raw
         tr-set-grad-enabled/raw
         tr-is-grad-enabled/raw
         tr-tensor-sub!/raw
         tr-tensor-zero!/raw
         tr-tensor-mul!/raw)

(define-torch tr-tensor-requires-grad!/raw
  (_fun (t : _Tensor) (requires? : _bool) -> _int)
  #:c-id tr_tensor_requires_grad_)

(define-torch tr-tensor-requires-grad/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out))))
  #:c-id tr_tensor_requires_grad)

;; Cheap predicate: no handle allocation, no tr_last_error side effect on
;; the "no gradient" path (unlike probing tr-tensor-grad/raw for NULL).
(define-torch tr-tensor-has-grad/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out))))
  #:c-id tr_tensor_has_grad)

(define-torch tr-tensor-backward/raw
  (_fun (t : _Tensor) -> _int)
  #:c-id tr_tensor_backward)

(define-torch tr-tensor-grad/raw
  (_fun (t : _Tensor) -> _Tensor/null)
  #:c-id tr_tensor_grad
  #:wrap tensor-allocator)

(define-torch tr-tensor-detach/raw
  (_fun (t : _Tensor) -> _Tensor/null)
  #:c-id tr_tensor_detach
  #:wrap tensor-allocator)

(define-torch tr-set-grad-enabled/raw
  (_fun (enabled? : _bool) -> _int)
  #:c-id tr_set_grad_enabled)

(define-torch tr-is-grad-enabled/raw
  (_fun (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out))))
  #:c-id tr_is_grad_enabled)

;; t -= alpha * other (ATen sub_'s alpha form), the SGD update primitive.
(define-torch tr-tensor-sub!/raw
  (_fun (t : _Tensor) (other : _Tensor) (alpha : _double) -> _int)
  #:c-id tr_tensor_sub_)

(define-torch tr-tensor-zero!/raw
  (_fun (t : _Tensor) -> _int)
  #:c-id tr_tensor_zero_)

(define-torch tr-tensor-mul!/raw
  (_fun (t : _Tensor) (value : _double) -> _int)
  #:c-id tr_tensor_mul_)
