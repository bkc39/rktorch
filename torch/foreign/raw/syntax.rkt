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
         tr-tensor-free/finalizer
         tr-tensor-free/checked
         guard-finalizer
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

;; The bare C binding, private: composed into the two public entry points
;; below, each carrying exactly the wrap its calling context needs.
(define-torch tr-tensor-free/unwrapped
  (_fun _Tensor -> _void)
  #:c-id tr_tensor_free)

;; The raising direct binding: explicit, synchronous release paths
;; (structs.rkt's tensor-free! via the unsafe submodule) call THIS — a
;; deliberate caller can and should see marshalling/release failures. The
;; (deallocator) wrap makes an explicit call CANCEL the pending GC
;; finalizer (structs.rkt's lifetime comment always claimed this; without
;; the wrap the finalizer stayed registered and later raised a freed-tag
;; marshalling error inside finalization — the #38 cascade class).
(define tr-tensor-free/checked
  ((deallocator) tr-tensor-free/unwrapped))

;; Wrap a release procedure for use as a GC-finalizer deallocator: swallow
;; exn:fail so nothing raises out of finalization. Inside GC finalization a
;; raised exception resurfaces from whatever code triggered the collection —
;; including the error display handlers, which then allocate, re-trigger GC,
;; hit the next poisoned handle, and loop (issue #38's "invalid memory
;; reference" cascade). Free failures split into two disjoint classes:
;;
;;  * A C++ throw during storage release terminates inside libtorch's own
;;    noexcept frames before ANY handler, C++ or Racket (pinned by
;;    cpp/tests/torchrkt/finalizer_death_test.cpp) — one clean process
;;    death; nothing at any layer can catch it, and this guard never runs.
;;  * Failures the Racket runtime itself observes and raises as exceptions
;;    at the finalizer boundary — e.g. faults converted to "invalid memory
;;    reference" — which are exactly the looping class reported in #38.
;;    Those ARE catchable, and only here: the guard swallows them so the
;;    error machinery never re-enters GC. A finalizer has nowhere to
;;    report; leaking one handle beats a cascade.
;;
;; The catch is TOTAL — every raised value, not just exn:fail —
;; because ffi/unsafe/alloc's contract for a deallocate argument is "a
;; function that never raises an exception", full stop: a bare raised
;; value or a break escaping here re-enters the cascade through the same
;; choke point. This runs on the runtime's finalizer thread, so
;; swallowing a break loses nothing from user threads.
;;
;; Exposed as a combinator (not baked into one binding) so the swallow
;; semantics are unit-testable and reusable by future finalizer bindings.
(define ((guard-finalizer release) t)
  (with-handlers ([(lambda (_) #t) void])
    (release t)))

;; The deallocator must be defined before any allocator that references it;
;; every tensor-returning binding wraps with (allocator tr-tensor-free/finalizer).
;; This name is the FINALIZER-context entry point only — explicit frees use
;; tr-tensor-free/checked above. Built on the UNWRAPPED binding, not
;; /checked: routing the finalizer through the (deallocator)-wrapped
;; function would run its cancel-my-own-registration step from inside the
;; very finalizer that registration refers to — a self-referential use of
;; the allocator machinery we avoid by construction rather than trust.
(define tr-tensor-free/finalizer (guard-finalizer tr-tensor-free/unwrapped))

;; --- op-definer macros -------------------------------------------------
;; The three uniform op shapes. Each expands to a define-torch binding whose
;; fresh handle is GC-managed via the allocator wrap.

(define-syntax (define-unary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (t : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap (allocator tr-tensor-free/finalizer))]))

(define-syntax (define-binary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap (allocator tr-tensor-free/finalizer))]))

(define-syntax (define-scalar/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _double) -> _Tensor/null)
         #:c-id c-id
         #:wrap (allocator tr-tensor-free/finalizer))]))

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
