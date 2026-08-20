#lang racket/base

;; The pure FFI-definer substrate every raw module builds on:
;;
;;  * define-torch — the core FFI-definer form, bound against libtorchrkt
;;  * the opaque _Tensor cpointer type (it lives here, with the definer,
;;    because every raw binding's _fun spec references it)
;;  * define-arith — the shadow-arithmetic generator used by
;;    foreign/operators.rkt; it takes the tensor predicate and ops as
;;    arguments so this module needs nothing from the layers above
;;
;; The tensor lifetime + #37 accounting layer (frees, ledger, probes,
;; tensor-allocator, and the op-definer macros that expand into it) lives
;; in memory.rkt, one module up the require chain.
;;
;; `native-libs-dir` is resolved relative to this file, which lives at
;; torch/foreign/raw/ — two directories below the collection root — so the
;; path to torch/native-libs/ climbs two levels.

(require (for-syntax racket/base
                     ;; whole-module on purpose: syntax-parse patterns
                     ;; reference many exported bindings
                     syntax/parse/pre)
         (only-in ffi/unsafe define-cpointer-type ffi-lib)
         (only-in ffi/unsafe/define define-ffi-definer)
         ;; whole-module on purpose: define-runtime-path expands into
         ;; phase-1 code that needs bindings (e.g. #%datum) the full require
         ;; re-exports for-syntax; only-in strips them.
         racket/runtime-path)

(provide define-torch
         _Tensor
         _Tensor/null ;; noqa
         Tensor? ;; noqa
         define-arith)

(define-runtime-path native-libs-dir "../../native-libs")

(define-ffi-definer define-torch
  (ffi-lib (build-path native-libs-dir "libtorchrkt")))

;; `define-cpointer-type` generates three names:
;;   _Tensor       — non-null cpointer type (tag 'Tensor)
;;   _Tensor/null  — nullable cpointer type (used for NULL-on-error returns)
;;   Tensor?       — predicate
(define-cpointer-type _Tensor)

;; --- shadow-arithmetic generator -----------------------------------------
;; (define-arith name tensor-pred tensor-op base-op unary-tensor) defines a
;; variadic op: a numeric fast path straight to base-op, unary forms
;; mirroring racket ((- t) negates via unary-tensor), and left-folding
;; chains where any tensor operand routes to tensor-op. The predicate and
;; ops arrive as arguments, so expansion resolves them at the use site and
;; this module stays independent of the op layer.

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
