#lang racket/base

(require (only-in ffi/unsafe _fun _int _int64 _ptr)
         (only-in "memory.rkt"
                  _tr-device-type
                  tensor-allocator
                  tr-cuda-empty-cache/raw
                  tr-mps-empty-cache/raw
                  tr-tensor-device/raw)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide _tr-device-type
         tr-cuda-is-available/raw
         tr-mps-is-available/raw
         tr-mps-empty-cache/raw
         tr-cuda-device-count/raw
         tr-cuda-empty-cache/raw
         tr-cuda-memory-stats/raw
         tr-set-default-device/raw
         tr-get-default-device/raw
         tr-tensor-to-device/raw
         tr-tensor-device/raw)

(define-torch tr-cuda-memory-stats/raw
  (_fun (index : _int64)
        (allocated : (_ptr o _int64))
        (reserved : (_ptr o _int64))
        (peak : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc allocated reserved peak))
  #:c-id tr_cuda_memory_stats)

(define-torch tr-cuda-is-available/raw
  (_fun -> _int)
  #:c-id tr_cuda_is_available)

(define-torch tr-cuda-device-count/raw
  (_fun -> _int)
  #:c-id tr_cuda_device_count)

(define-torch tr-mps-is-available/raw
  (_fun -> _int)
  #:c-id tr_mps_is_available)

(define-torch tr-set-default-device/raw
  (_fun (type : _tr-device-type) (index : _int64) -> _int)
  #:c-id tr_set_default_device)

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

