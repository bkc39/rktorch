#lang racket/base

;; Raw device APIs (device.h): the CUDA availability queries, the process-wide
;; default device the constructors honor, and per-tensor device moves. The
;; get/query functions follow the integer-status, out-parameter contract; the
;; move allocates a fresh handle, so it carries the GC allocator wrap.

(require (only-in ffi/unsafe _fun _int _int64 _ptr)
         (only-in "memory.rkt"
                  _tr-device-type
                  tensor-allocator
                  tr-tensor-device/raw)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide _tr-device-type
         tr-cuda-is-available/raw
         tr-cuda-device-count/raw
         tr-set-default-device/raw
         tr-get-default-device/raw
         tr-tensor-to-device/raw
         tr-tensor-device/raw)

;; _tr-device-type and tr-tensor-device/raw live in syntax.rkt (the #37
;; accounting there needs them too, and this module requires syntax.rkt, so
;; that is the cycle-free canonical home); re-provided above unchanged.

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
  #:wrap tensor-allocator)

