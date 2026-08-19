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
         (only-in ffi/unsafe
                  _double _enum _fun _int _int64 _ptr _void
                  define-cpointer-type ffi-lib)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in ffi/unsafe/define define-ffi-definer)
         ;; whole-module on purpose: define-runtime-path expands into
         ;; phase-1 code that needs bindings (e.g. #%datum) the full require
         ;; re-exports for-syntax; only-in strips them.
         racket/runtime-path
         (only-in "../device-type.rkt" device device-index device-type))

(provide define-torch
         _Tensor
         _Tensor/null ;; noqa
         Tensor? ;; noqa
         tr-tensor-free/finalizer
         tr-tensor-free/checked
         guard-finalizer
         tensor-allocator
         native-memory-use
         _tr-device-type ;; noqa
         tr-tensor-device/raw
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
;; unaccount! (defined with the #37 ledger below) drops the handle's
;; pressure charge before the release — a deliberate explicit free must
;; leave the ledger as promptly as it leaves the allocator registration.
(define tr-tensor-free/checked
  (let ([release ((deallocator) tr-tensor-free/unwrapped)])
    (lambda (t)
      (unaccount! t)
      (release t))))

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

;; --- native-memory accounting (#37) --------------------------------------
;; Racket's GC sees a tensor as a tiny wrapper, so native buffers pile up
;; awaiting finalizers nothing prompts it to run (measured: a 2 GB churn of
;; dropped tensors retained every byte). Each allocation therefore charges
;; a phantom-bytes object — Racket's purpose-built external-allocation
;; accounting — sized to the tensor's nbytes, so collection scheduling
;; scales with native usage and finalizers run in proportion.
;;
;; The ledger is a weak-keyed side table: handle -> allocation record. Weak
;; keys make it self-healing (a missed unaccount! drops with the handle and
;; the phantom collects on its own). Per-device totals are FOLDED ON QUERY,
;; never maintained incrementally: every hot-path operation is a single
;; per-key hash-set!/remove!, leaving no read-modify-write to lose updates
;; across green-thread yield points (finalizers run on the runtime's
;; executor thread, so every program is effectively multi-threaded here).
;;
;; The ledger reports handle-attributed bytes — the view's extent, the
;; signal GC pressure needs — NOT total device usage: views over-count
;; shared storage, and ATen-internal buffers never cross this boundary.

;; The per-handle entry. A named record, never an anonymous tuple: it is
;; also the extension point later legs (typed-OOM retry, failure counters)
;; hang fields on.
(struct allocation (phantom nbytes device))

(define allocations (make-weak-hasheq))

;; Serializes ledger mutation against the query fold. account!/unaccount!
;; run on user threads AND the finalizer context, and a hash-remove!
;; landing between iterator steps of the fold (thread swaps can occur at
;; allocation points inside it) would invalidate the iteration. The
;; critical sections are guarded with ATOMIC MODE, not a semaphore:
;; finalizers already run in atomic mode, where blocking on a semaphore
;; is an internal error ("attempt to deschedule the current thread in
;; atomic mode") — call-as-atomic is reentrant there and never blocks,
;; and within a place atomic mode excludes all other green threads.
(define (call-with-ledger thunk)
  (call-as-atomic thunk))

;; Probes shared with raw/device.rkt, which requires this module — making
;; here the cycle-free canonical home for the device enum + query binding
;; (the #37 accounting below needs them too).
(define-torch tr-tensor-nbytes/raw
  (_fun _Tensor (out : (_ptr o _int64)) -> (rc : _int) -> (values rc out))
  #:c-id tr_tensor_nbytes)

;; Mirrors the tr_device_type C enum (device.h); int-width, like _tr-dtype.
(define _tr-device-type
  (_enum '(cpu = 0 cuda = 1)))

(define-torch tr-tensor-device/raw
  (_fun _Tensor
        (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_tensor_device)

;; Charge a fresh handle to the GC and record it in the ledger. Guarded
;; and best-effort: accounting must never fail an allocation that
;; succeeded (make-phantom-bytes itself can raise under memory pressure),
;; so any exn:fail here simply skips the charge. The predicate is
;; exn:fail?, NOT a total catch: this runs on the calling USER thread,
;; where swallowing exn:break would silently discard a cancellation —
;; breaks propagate. (The finalizer path stays totally guarded by
;; guard-finalizer's own catch around everything, unaccount! included.)
(define (account! t)
  (with-handlers ([exn:fail? void])
    (define-values (nb-rc nbytes) (tr-tensor-nbytes/raw t))
    (define-values (dev-rc type index) (tr-tensor-device/raw t))
    (when (and (zero? nb-rc) (zero? dev-rc))
      (define entry
        (allocation (make-phantom-bytes nbytes)
                    nbytes
                    (device type (if (eq? type 'cpu) 0 index))))
      (call-with-ledger (lambda () (hash-set! allocations t entry))))))

;; Drop a handle's ledger entry and release its charged pressure NOW
;; (set-phantom-bytes! to 0 rather than waiting for the phantom's own
;; collection). Guarded like account! (exn:fail? only — breaks propagate
;; on the explicit user-thread path; the finalizer path's guard-finalizer
;; supplies the total catch): bookkeeping must never turn a free into a
;; failure.
(define (unaccount! t)
  (with-handlers ([exn:fail? void])
    (call-with-ledger
     (lambda ()
       (define a (hash-ref allocations t #f))
       (when a
         (set-phantom-bytes! (allocation-phantom a) 0)
         (hash-remove! allocations t))))))

;; Live handle-attributed native bytes per device, folded from the ledger
;; at query time (see the design comment above): an alist of device struct
;; to byte total, cpu first, then cuda by ordinal. A query racing heavy
;; allocation sees an approximate snapshot — correct for a gauge.
(define (native-memory-use)
  (define totals (make-hash))
  (call-with-ledger
   (lambda ()
     (for ([a (in-hash-values allocations)])
       (hash-update! totals (allocation-device a)
                     (lambda (n) (+ n (allocation-nbytes a)))
                     0))))
  (sort (hash->list totals)
        (lambda (x y)
          (define dx (car x))
          (define dy (car y))
          (cond
            [(eq? (device-type dx) (device-type dy))
             (< (device-index dx) (device-index dy))]
            [else (eq? (device-type dx) 'cpu)]))))

;; The deallocator must be defined before any allocator that references it;
;; every tensor-returning binding wraps with `tensor-allocator` below.
;; This name is the FINALIZER-context entry point only — explicit frees use
;; tr-tensor-free/checked above. Built on the UNWRAPPED binding, not
;; /checked: routing the finalizer through the (deallocator)-wrapped
;; function would run its cancel-my-own-registration step from inside the
;; very finalizer that registration refers to — a self-referential use of
;; the allocator machinery we avoid by construction rather than trust.
(define tr-tensor-free/finalizer
  (guard-finalizer
   (lambda (t)
     (unaccount! t)
     (tr-tensor-free/unwrapped t))))

;; The one wrap every tensor-returning raw binding carries (hand-written
;; via the op-definer macros below and the explicit #:wrap sites; generated
;; via define-generated.rkt): GC-managed lifetime via the allocator, plus
;; the #37 pressure charge on each fresh handle. NULL (error) results are
;; never charged.
(define (tensor-allocator raw-fn)
  (define wrapped ((allocator tr-tensor-free/finalizer) raw-fn))
  (lambda args
    (define t (apply wrapped args))
    (when t (account! t))
    t))

;; --- op-definer macros -------------------------------------------------
;; The three uniform op shapes. Each expands to a define-torch binding whose
;; fresh handle is GC-managed via the allocator wrap.

(define-syntax (define-unary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (t : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

(define-syntax (define-binary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

(define-syntax (define-scalar/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _double) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

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
