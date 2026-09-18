#lang racket/base

(require (only-in ffi/unsafe _fun _int _int64 _ptr _string/utf-8)
         (only-in "../device-type.rkt" device-index device-type)
         (only-in "memory.rkt"
                  _tr-device-type
                  tensor-allocator
                  tr-cuda-empty-cache/raw
                  tr-mps-empty-cache/raw
                  tr-tensor-device/raw)
         (only-in "pressure.rkt" install-device-queries!)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch)
         (only-in "tensor.rkt" _tr-dtype))

(provide _tr-device-type
         tr-cuda-is-available/raw
         tr-mps-is-available/raw
         tr-mps-empty-cache/raw
         tr-cuda-device-count/raw
         tr-cuda-empty-cache/raw
         tr-cuda-mem-get-info/raw
         tr-cuda-memory-stats/raw
         tr-cuda-reset-peak-stats/raw
         tr-cuda-set-allocator-settings/raw
         tr-mps-memory-info/raw
         tr-set-default-device/raw
         tr-get-default-device/raw
         tr-tensor-to-device/raw
         tr-tensor-to/raw
         tr-tensor-to!/raw
         tr-tensor-device/raw)

(define-torch tr-cuda-mem-get-info/raw
  (_fun (index : _int64)
        (free : (_ptr o _int64))
        (total : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc free total))
  #:c-id tr_cuda_mem_get_info)

(define-torch tr-cuda-memory-stats/raw
  (_fun (index : _int64)
        (allocated : (_ptr o _int64))
        (reserved : (_ptr o _int64))
        (peak : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc allocated reserved peak))
  #:c-id tr_cuda_memory_stats)

(define-torch tr-mps-memory-info/raw
  (_fun (allocated : (_ptr o _int64))
        (driver : (_ptr o _int64))
        (recommended : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc allocated driver recommended))
  #:c-id tr_mps_memory_info)

;; the pressure engine sits below this module, so the readings it needs are
;; handed down; #f is "this device cannot say"
(define (device-capacity dev)
  (case (device-type dev)
    [(cuda)
     (define-values (rc _free total)
       (tr-cuda-mem-get-info/raw (device-index dev)))
     (and (zero? rc) total)]
    [(mps)
     ;; the working set Metal recommends staying under, 0 when absent
     (define-values (rc _allocated _driver recommended)
       (tr-mps-memory-info/raw))
     (and (zero? rc) (positive? recommended) recommended)]
    [else #f]))

(define (device-allocated dev)
  (case (device-type dev)
    [(cuda)
     (define-values (rc allocated _reserved _peak)
       (tr-cuda-memory-stats/raw (device-index dev)))
     (and (zero? rc) allocated)]
    [(mps)
     (define-values (rc allocated _driver _recommended)
       (tr-mps-memory-info/raw))
     (and (zero? rc) allocated)]
    [else #f]))

(install-device-queries! #:capacity device-capacity
                         #:allocated device-allocated)

(define-torch tr-cuda-reset-peak-stats/raw
  (_fun (index : _int64) -> _int)
  #:c-id tr_cuda_reset_peak_stats)

(define-torch tr-cuda-set-allocator-settings/raw
  (_fun (settings : _string/utf-8) -> _int)
  #:c-id tr_cuda_set_allocator_settings)

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

(define-torch tr-tensor-to/raw
  (_fun (t : _Tensor)
        (type : _tr-device-type)
        (index : _int64)
        (dtype : _tr-dtype)
        -> _Tensor/null)
  #:c-id tr_tensor_to
  #:wrap tensor-allocator)

(define-torch tr-tensor-to!/raw
  (_fun (t : _Tensor)
        (type : _tr-device-type)
        (index : _int64)
        (dtype : _tr-dtype)
        -> _int)
  #:c-id tr_tensor_to_)

