#lang racket/base

;; The pure FFI-definer substrate every raw module builds on: define-torch,
;; the opaque _Tensor cpointer type, and the define-arith shadow-arithmetic
;; generator. Lifetime + accounting live one module up, in memory.rkt.

(require (for-syntax racket/base
                     ;; whole-module: syntax-parse patterns reference many
                     ;; of its bindings
                     syntax/parse/pre)
         (only-in ffi/unsafe define-cpointer-type ffi-lib)
         (only-in ffi/unsafe/define define-ffi-definer)
         ;; whole-module: define-runtime-path expands into phase-1 code
         ;; needing bindings the full require re-exports; only-in strips them.
         racket/runtime-path)

(provide define-torch
         _Tensor
         _Tensor/null ;; noqa
         Tensor? ;; noqa
         define-arith)

(define-runtime-path native-libs-dir "../../native-libs")

(define-ffi-definer define-torch
  (ffi-lib (build-path native-libs-dir "libtorchrkt")))

;; Generates _Tensor, _Tensor/null (for NULL-on-error returns), and the
;; Tensor? predicate.
(define-cpointer-type _Tensor)

;; Defines a variadic shadow-arithmetic op: numeric fast path to base-op,
;; unary forms mirroring racket, left-folding chains routing any tensor
;; operand to tensor-op. The predicate and ops arrive as arguments so this
;; module stays independent of the op layer.
(define-syntax (define-arith stx)
  (syntax-parse stx
    [(_ name:id tensor-pred:expr tensor-op:expr base-op:expr
        unary-tensor:expr)
     #'(define (name . args)
         (cond
           ;; (andmap number? '()) is #t, so this also covers (+) => 0.
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
