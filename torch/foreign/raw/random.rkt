#lang racket/base

;; Everything here draws the global RNG stream: no-retry wrap only.

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

(define-torch tr-tensor-uniform!/raw
  (_fun (t : _Tensor) (low : _double) (high : _double) -> _int)
  #:c-id tr_tensor_uniform_)
