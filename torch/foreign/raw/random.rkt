#lang racket/base

;; Raw RNG tensor constructors. These draw from the global RNG stream, so
;; they take the NO-RETRY wrap (tensor-allocator/rng — see memory.rkt).

(require (only-in ffi/unsafe _double _fun _int _int64)
         (only-in ffi/vector _s64vector)
         (only-in "memory.rkt" tensor-allocator/rng)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-randn/raw
         tr-rand/raw
         tr-tensor-uniform!/raw)

(define-torch tr-randn/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_randn
  #:wrap tensor-allocator/rng)

(define-torch tr-rand/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_rand
  #:wrap tensor-allocator/rng)

;; In-place fill on [low, high); consumes the global RNG exactly like
;; torch.Tensor.uniform_, which nn init parity depends on.
(define-torch tr-tensor-uniform!/raw
  (_fun (t : _Tensor) (low : _double) (high : _double) -> _int)
  #:c-id tr_tensor_uniform_)
