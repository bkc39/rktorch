#lang racket/base

;; Tensor draws take the global stream unless handed a generator: no-retry
;; wrap only, since a retried draw would advance the stream.

(require (only-in ffi/unsafe
                  _double _fun _int _int64 _ptr _uint64 _void
                  define-cpointer-type make-ctype)
         (only-in ffi/unsafe/alloc allocator)
         (only-in ffi/vector _s64vector)
         (only-in "memory.rkt"
                  _tr-device-type swallow-and-count-failure tensor-allocator/no-retry)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch)
         (only-in "tensor.rkt" _tr-dtype))

(provide Generator? ;; noqa
         _Generator/current
         _Generator/null
         current-generator/raw
         tr-generator-device/raw
         tr-generator-draw-seed/raw
         tr-generator-new-on/raw
         tr-randn/raw
         tr-randn-gen-on/raw
         tr-rand/raw
         tr-rand-gen-on/raw
         tr-randperm/raw
         tr-tensor-normal!/raw
         tr-tensor-uniform!/raw)

(define-cpointer-type _Generator)

(define current-generator/raw (make-parameter #f))

(define _Generator/current
  (make-ctype _Generator/null (lambda (g) (or g (current-generator/raw))) #f))

(define-torch tr-generator-free/unwrapped
  (_fun _Generator -> _void)
  #:c-id tr_generator_free)

;; a generator is a handle but not a tensor: no bytes to charge, so the
;; guarded finalizer alone releases it
(define-torch tr-generator-new-on/raw
  (_fun (seed : _uint64) (type : _tr-device-type) (index : _int64)
        -> _Generator/null)
  #:c-id tr_generator_new_on
  #:wrap (allocator (swallow-and-count-failure tr-generator-free/unwrapped)))

(define-torch tr-generator-device/raw
  (_fun (generator : _Generator)
        (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_generator_device)

(define-torch tr-generator-draw-seed/raw
  (_fun (generator : _Generator/current) (out : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id tr_generator_draw_seed)

;; a draw from either stream; no retry, as for the other RNG bindings
(define-torch tr-randperm/raw
  (_fun (n : _int64) (generator : _Generator/current) -> _Tensor/null)
  #:c-id tr_randperm
  #:wrap tensor-allocator/no-retry)

(define-torch tr-randn-gen-on/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        (type : _tr-device-type)
        (index : _int64)
        (dtype : _tr-dtype)
        (generator : _Generator/current)
        -> _Tensor/null)
  #:c-id tr_randn_gen_on
  #:wrap tensor-allocator/no-retry)

(define-torch tr-rand-gen-on/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        (type : _tr-device-type)
        (index : _int64)
        (dtype : _tr-dtype)
        (generator : _Generator/current)
        -> _Tensor/null)
  #:c-id tr_rand_gen_on
  #:wrap tensor-allocator/no-retry)

(define-torch tr-randn/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_randn
  #:wrap tensor-allocator/no-retry)

(define-torch tr-rand/raw
  (_fun (dims : (_s64vector i))
        (ndim : _int64)
        -> _Tensor/null)
  #:c-id tr_rand
  #:wrap tensor-allocator/no-retry)

(define-torch tr-tensor-uniform!/raw
  (_fun (t : _Tensor) (low : _double) (high : _double)
        (generator : _Generator/current)
        -> _int)
  #:c-id tr_tensor_uniform_)

(define-torch tr-tensor-normal!/raw
  (_fun (t : _Tensor) (mean : _double) (std : _double)
        (generator : _Generator/current)
        -> _int)
  #:c-id tr_tensor_normal_)
