#lang racket/base

;; The FFI substrate every raw module builds on:
;;
;;  * define-torch — the core FFI-definer form, bound against libtorchrkt
;;  * the opaque _Tensor cpointer type and its deallocator (they live here,
;;    with the definer, because the op-definer macros below expand into
;;    references to them)
;;  * the op-definer macros for the three uniform op shapes (unary, binary,
;;    tensor-scalar), shared by the elementwise/linalg/reduce families
;;  * define-arith — the shadow-arithmetic generator used by
;;    foreign/operators.rkt; it takes the tensor predicate and ops as
;;    arguments so this module needs nothing from the layers above
;;
;; `native-libs-dir` is resolved relative to this file, which lives at
;; torch/foreign/raw/ — two directories below the collection root — so the
;; path to torch/native-libs/ climbs two levels.

(require (for-syntax racket/base
                     ;; whole-module on purpose: syntax-parse patterns
                     ;; reference many exported bindings
                     syntax/parse/pre)
         (only-in ffi/unsafe _double _fun _void define-cpointer-type ffi-lib)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/define define-ffi-definer)
         ;; whole-module on purpose: define-runtime-path expands into
         ;; phase-1 code that needs bindings (e.g. #%datum) the full require
         ;; re-exports for-syntax; only-in strips them.
         racket/runtime-path)

(provide define-torch
         _Tensor
         _Tensor/null ;; noqa
         Tensor? ;; noqa
         tr-tensor-free/raw
         define-unary/raw
         define-binary/raw
         define-scalar/raw
         define-arith)

(define-runtime-path native-libs-dir "../../native-libs")

(define-ffi-definer define-torch
  (ffi-lib (build-path native-libs-dir "libtorchrkt")))

;; `define-cpointer-type` generates three names:
;;   _Tensor       — non-null cpointer type (tag 'Tensor)
;;   _Tensor/null  — nullable cpointer type (used for NULL-on-error returns)
;;   Tensor?       — predicate
(define-cpointer-type _Tensor)

;; The deallocator must be defined before any allocator that references it;
;; every tensor-returning binding wraps with (allocator tr-tensor-free/raw).
(define-torch tr-tensor-free/raw
  (_fun _Tensor -> _void)
  #:c-id tr_tensor_free)

;; --- op-definer macros -------------------------------------------------
;; The three uniform op shapes. Each expands to a define-torch binding whose
;; fresh handle is GC-managed via the allocator wrap.

(define-syntax (define-unary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (t : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap (allocator tr-tensor-free/raw))]))

(define-syntax (define-binary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap (allocator tr-tensor-free/raw))]))

(define-syntax (define-scalar/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _double) -> _Tensor/null)
         #:c-id c-id
         #:wrap (allocator tr-tensor-free/raw))]))

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
           [(andmap number? args) (apply base-op args)]
           [(null? args) (base-op)]
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
