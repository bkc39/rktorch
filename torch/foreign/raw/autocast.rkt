#lang racket/base

(require (only-in ffi/unsafe _bool _fun _int _ptr)
         (only-in "memory.rkt" _tr-device-type)
         (only-in "syntax.rkt" define-torch)
         (only-in "tensor.rkt" _tr-dtype dtype-code->symbol))

(provide tr-set-autocast-enabled/raw
         tr-is-autocast-enabled/raw
         tr-autocast-dtype/raw)

(define-torch tr-set-autocast-enabled/raw
  (_fun (type : _tr-device-type) (dtype : _tr-dtype) (enabled? : _bool) -> _int)
  #:c-id tr_set_autocast_enabled)

(define-torch tr-is-autocast-enabled/raw
  (_fun (type : _tr-device-type)
        (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (not (zero? out))))
  #:c-id tr_is_autocast_enabled)

;; out is a plain _int: on the error path the C side never writes it (see
;; tr-tensor-dtype/raw)
(define-torch tr-autocast-dtype/raw
  (_fun (type : _tr-device-type)
        (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc (dtype-code->symbol out)))
  #:c-id tr_autocast_dtype)
