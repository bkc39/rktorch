#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe _double _fun)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide define-binary/raw
         define-scalar/raw
         define-unary/raw)

(define-syntax-parse-rule (define-unary/raw name:id c-id:id)
  (define-torch name
    (_fun (t : _Tensor) -> _Tensor/null)
    #:c-id c-id
    #:wrap tensor-allocator))

(define-syntax-parse-rule (define-binary/raw name:id c-id:id)
  (define-torch name
    (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
    #:c-id c-id
    #:wrap tensor-allocator))

(define-syntax-parse-rule (define-scalar/raw name:id c-id:id)
  (define-torch name
    (_fun (a : _Tensor) (b : _double) -> _Tensor/null)
    #:c-id c-id
    #:wrap tensor-allocator))
