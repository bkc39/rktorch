#lang racket/base

(require (only-in ffi/unsafe _bytes _fun _int32 _int64)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor/null define-torch))

(provide tr-image-decode/raw)

(define-torch tr-image-decode/raw
  (_fun (data : _bytes)
        (len : _int64)
        (channels : _int32)
        -> _Tensor/null)
  #:c-id tr_image_decode
  #:wrap tensor-allocator)
