#lang racket/base

(require (only-in ffi/unsafe _bool _fun _int _ptr _string/utf-8 _uint64)
         (only-in "memory.rkt" tr-last-error-kind/raw)
         (only-in "syntax.rkt" define-torch))

(provide tr-version/raw
         tr-last-error/raw
         tr-last-error-kind/raw
         tr-manual-seed/raw
         tr-set-deterministic-algorithms/raw
         tr-get-deterministic-algorithms/raw
         tr-get-deterministic-algorithms-warn-only/raw
         tr-set-cudnn-deterministic/raw
         tr-get-cudnn-deterministic/raw
         tr-set-cudnn-benchmark/raw
         tr-get-cudnn-benchmark/raw
         tr-set-float32-matmul-precision/raw
         tr-get-float32-matmul-precision/raw
         tr-set-cuda-matmul-allow-tf32/raw
         tr-get-cuda-matmul-allow-tf32/raw
         tr-set-cudnn-allow-tf32/raw
         tr-get-cudnn-allow-tf32/raw)

(define-torch tr-version/raw
  (_fun -> _string/utf-8)
  #:c-id tr_version)

(define-torch tr-last-error/raw
  (_fun -> _string/utf-8)
  #:c-id tr_last_error)

(define-torch tr-manual-seed/raw
  (_fun _uint64 -> _int)
  #:c-id tr_manual_seed)

(define _flag-setter
  (_fun (on? : _bool) -> _int))

(define _flag-getter
  (_fun (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out)))))

(define-torch tr-set-deterministic-algorithms/raw
  (_fun (on? : _bool) (warn-only? : _bool) -> _int)
  #:c-id tr_set_deterministic_algorithms)

(define-torch tr-get-deterministic-algorithms/raw
  _flag-getter
  #:c-id tr_get_deterministic_algorithms)

(define-torch tr-get-deterministic-algorithms-warn-only/raw
  _flag-getter
  #:c-id tr_get_deterministic_algorithms_warn_only)

(define-torch tr-set-cudnn-deterministic/raw
  _flag-setter
  #:c-id tr_set_cudnn_deterministic)

(define-torch tr-get-cudnn-deterministic/raw
  _flag-getter
  #:c-id tr_get_cudnn_deterministic)

(define-torch tr-set-cudnn-benchmark/raw
  _flag-setter
  #:c-id tr_set_cudnn_benchmark)

(define-torch tr-get-cudnn-benchmark/raw
  _flag-getter
  #:c-id tr_get_cudnn_benchmark)

(define-torch tr-set-float32-matmul-precision/raw
  (_fun (code : _int) -> _int)
  #:c-id tr_set_float32_matmul_precision)

(define-torch tr-get-float32-matmul-precision/raw
  (_fun (out : (_ptr o _int)) -> (rc : _int) -> (values rc out))
  #:c-id tr_get_float32_matmul_precision)

(define-torch tr-set-cuda-matmul-allow-tf32/raw
  _flag-setter
  #:c-id tr_set_cuda_matmul_allow_tf32)

(define-torch tr-get-cuda-matmul-allow-tf32/raw
  _flag-getter
  #:c-id tr_get_cuda_matmul_allow_tf32)

(define-torch tr-set-cudnn-allow-tf32/raw
  _flag-setter
  #:c-id tr_set_cudnn_allow_tf32)

(define-torch tr-get-cudnn-allow-tf32/raw
  _flag-getter
  #:c-id tr_get_cudnn_allow_tf32)
