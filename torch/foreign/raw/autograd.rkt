#lang racket/base

;; Raw autograd bindings (autograd.h). tr-tensor-grad/raw returns a handle
;; that SHARES storage with the live .grad — it still gets its own allocator
;; finalizer because the C side wraps it in a fresh tr_tensor box; freeing the
;; box releases one reference, not the gradient itself.

(require (only-in ffi/unsafe _bool _double _fun _int _ptr)
         (only-in ffi/unsafe/alloc allocator)
         (only-in "syntax.rkt" define-torchrkt)
         (only-in "tensor.rkt" _Tensor _Tensor/null tr-tensor-free/raw))

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

(define-torchrkt tr-tensor-requires-grad!/raw
  (_fun (t : _Tensor) (requires? : _bool) -> _int)
  #:c-id tr_tensor_requires_grad_)

(define-torchrkt tr-tensor-requires-grad/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out))))
  #:c-id tr_tensor_requires_grad)

;; Cheap predicate: no handle allocation, no tr_last_error side effect on
;; the "no gradient" path (unlike probing tr-tensor-grad/raw for NULL).
(define-torchrkt tr-tensor-has-grad/raw
  (_fun (t : _Tensor)
        (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out))))
  #:c-id tr_tensor_has_grad)

(define-torchrkt tr-tensor-backward/raw
  (_fun (t : _Tensor) -> _int)
  #:c-id tr_tensor_backward)

(define-torchrkt tr-tensor-grad/raw
  (_fun (t : _Tensor) -> _Tensor/null)
  #:c-id tr_tensor_grad
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-tensor-detach/raw
  (_fun (t : _Tensor) -> _Tensor/null)
  #:c-id tr_tensor_detach
  #:wrap (allocator tr-tensor-free/raw))

(define-torchrkt tr-set-grad-enabled/raw
  (_fun (enabled? : _bool) -> _int)
  #:c-id tr_set_grad_enabled)

(define-torchrkt tr-is-grad-enabled/raw
  (_fun (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out))))
  #:c-id tr_is_grad_enabled)

;; t -= alpha * other (ATen sub_'s alpha form), the SGD update primitive.
(define-torchrkt tr-tensor-sub!/raw
  (_fun (t : _Tensor) (other : _Tensor) (alpha : _double) -> _int)
  #:c-id tr_tensor_sub_)

(define-torchrkt tr-tensor-zero!/raw
  (_fun (t : _Tensor) -> _int)
  #:c-id tr_tensor_zero_)

(define-torchrkt tr-tensor-mul!/raw
  (_fun (t : _Tensor) (value : _double) -> _int)
  #:c-id tr_tensor_mul_)
