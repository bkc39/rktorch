#lang racket/base

(require (only-in "error.rkt" check-ok)
         (only-in "raw/tensor.rkt"
                  tr-tensor-data-ptr/raw tr-tensor-storage-ptr/raw))

(provide tensor-data-ptr
         tensor-storage-ptr)

(define ((address-reader raw who) t)
  (define-values (rc address) (raw t))
  (check-ok rc who)
  address)

(define tensor-data-ptr
  (address-reader tr-tensor-data-ptr/raw 'tensor-data-ptr))

(define tensor-storage-ptr
  (address-reader tr-tensor-storage-ptr/raw 'tensor-storage-ptr))
