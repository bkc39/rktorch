#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in ffi/unsafe define-cpointer-type)
         (only-in ffi/unsafe/define define-ffi-definer)
         (only-in "fault.rkt" latched)
         (only-in "native-library.rkt" native-library))

(provide define-torch
         _Tensor
         _Tensor/null ;; noqa
         Tensor? ;; noqa
         define-arith)

(define-ffi-definer define-torch/ffi native-library) ;; noqa

(define ((latched-under wrap) f) ;; noqa
  (wrap (latched f)))

(define-syntax (define-torch stx)
  (syntax-parse stx
    [(_ name:id type:expr
        (~alt (~optional (~seq #:wrap wrap:expr))
              (~seq option:keyword value:expr))
        ...)
     #'(define-torch/ffi name type
         #:wrap (latched-under (~? wrap values))
         (~@ option value) ...)]))

(define-cpointer-type _Tensor)

(define-syntax (define-arith stx)
  (syntax-parse stx
    [(_ name:id tensor-pred:expr tensor-op:expr base-op:expr
        unary-tensor:expr)
     #'(define (name . args)
         (cond
           [(andmap number? args) (apply base-op args)]
           [(null? (cdr args))
            (let ([a (car args)])
              (if (tensor-pred a) (unary-tensor a) (base-op a)))]
           [else
            (foldl (lambda (b acc)
                     (if (or (tensor-pred acc) (tensor-pred b))
                         (tensor-op acc b)
                         (base-op acc b)))
                   (car args)
                   (cdr args))]))]))
