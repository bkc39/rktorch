#lang racket/base

;; Tensor draws take the global stream unless handed a generator: no-retry
;; wrap only, since a retried draw would advance the stream.

(require (only-in ffi/unsafe
                  _double _fun _int _int64 _ptr _uint64 _void
                  define-cpointer-type)
         (only-in ffi/unsafe/alloc allocator)
         (only-in ffi/vector _s64vector)
         (only-in "memory.rkt"
                  _tr-device-type swallow-and-count-failure tensor-allocator/rng)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch)
         (only-in "tensor.rkt" _tr-dtype))

(provide Generator? ;; noqa
         tr-generator-draw-seed/raw
         tr-generator-new/raw
         tr-randn/raw
         tr-randn-on/raw
         tr-rand/raw
         tr-rand-on/raw
         tr-randperm/raw
         tr-tensor-uniform!/raw)

(define-cpointer-type _Generator)

(define-torch tr-generator-free/unwrapped
  (_fun _Generator -> _void)
  #:c-id tr_generator_free)

;; a generator is a handle but not a tensor: no bytes to charge, so the
;; guarded finalizer alone releases it
(define-torch tr-generator-new/raw
  (_fun (seed : _uint64) -> _Generator/null)
  #:c-id tr_generator_new
  #:wrap (allocator (swallow-and-count-failure tr-generator-free/unwrapped)))

(define-torch tr-generator-draw-seed/raw
  (_fun (generator : _Generator/null) (out : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_generator_draw_seed)

;; a draw from either stream; no retry, as for the other RNG bindings
(define-torch tr-randperm/raw
  (_fun (n : _int64) (generator : _Generator/null) -> _Tensor/null)
  #:c-id tr_randperm
  #:wrap tensor-allocator/rng)

(define-torch tr-randn-on/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        (type : _tr-device-type)
        (index : _int64)
        (dtype : _tr-dtype)
        -> _Tensor/null)
  #:c-id tr_randn_on
  #:wrap tensor-allocator/rng)

(define-torch tr-rand-on/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        (type : _tr-device-type)
        (index : _int64)
        (dtype : _tr-dtype)
        -> _Tensor/null)
  #:c-id tr_rand_on
  #:wrap tensor-allocator/rng)

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
