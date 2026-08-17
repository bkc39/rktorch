#lang racket/base

;; Raw device APIs (device.h): the CUDA availability queries, the process-wide
;; default device the constructors honor, and per-tensor device moves. The
;; get/query functions follow the integer-status, out-parameter contract; the
;; move allocates a fresh handle, so it carries the GC allocator wrap.

(require (only-in ffi/unsafe _enum _fun _int _int64 _ptr)
         (only-in ffi/unsafe/alloc allocator)
         (only-in "syntax.rkt"
                  _Tensor
                  _Tensor/null
                  define-torch
                  tr-tensor-free/finalizer))

(provide _tr-device-type
         tr-cuda-is-available/raw
         tr-cuda-device-count/raw
         tr-set-default-device/raw
         tr-get-default-device/raw
         tr-tensor-to-device/raw
         tr-tensor-device/raw)

;; Mirrors the tr_device_type C enum (device.h); int-width, like _tr-dtype.
(define _tr-device-type
  (_enum '(cpu = 0 cuda = 1)))

(define-torch tr-cuda-is-available/raw
  (_fun -> _int)
  #:c-id tr_cuda_is_available)

(define-torch tr-cuda-device-count/raw
  (_fun -> _int)
  #:c-id tr_cuda_device_count)

(define-torch tr-set-default-device/raw
  (_fun (type : _tr-device-type) (index : _int64) -> _int)
  #:c-id tr_set_default_device)

;; (raw) -> (values rc type index): the current default device.
(define-torch tr-get-default-device/raw
  (_fun (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_get_default_device)

(define-torch tr-tensor-to-device/raw
  (_fun (t : _Tensor) (type : _tr-device-type) (index : _int64) -> _Tensor/null)
  #:c-id tr_tensor_to_device
  #:wrap (allocator tr-tensor-free/finalizer))

;; (raw t) -> (values rc type index): the device the tensor lives on.
(define-torch tr-tensor-device/raw
  (_fun (t : _Tensor)
        (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_tensor_device)
