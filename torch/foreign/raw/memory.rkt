#lang racket/base

;; The tensor lifetime + native-memory substrate (extracted from syntax.rkt
;; per the plan's leg-1.5 refactor note, so syntax.rkt stays the pure FFI
;; definer it advertises itself as):
;;
;;  * the free-path split: the private unwrapped C binding composed into
;;    tr-tensor-free/checked (raising, finalizer-cancelling — explicit
;;    frees) and tr-tensor-free/finalizer (guarded — GC context only)
;;  * the #37 pressure ledger: phantom-bytes accounting, the weak side
;;    table, and the fold-on-query native-memory-use gauge
;;  * the probes those need (tr_tensor_nbytes / tr_tensor_device — this is
;;    their cycle-free canonical home; raw/device.rkt re-provides)
;;  * tensor-allocator — the one wrap every tensor-returning binding
;;    carries — and the op-definer macros that expand into it

(require (for-syntax racket/base
                     ;; whole-module on purpose: syntax-parse patterns
                     ;; reference many exported bindings
                     syntax/parse/pre)
         (only-in ffi/unsafe
                  _double _enum _fun _int _int64 _ptr _void
                  register-finalizer)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in "../device-type.rkt" device device-index device-type)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-tensor-free/finalizer
         tr-tensor-free/checked
         guard-finalizer
         finalizer-failures
         tensor-allocator
         tensor-allocator/rng
         oom-retry
         tr-cuda-empty-cache/raw
         tr-last-error-kind/raw
         native-memory-use
         _tr-device-type ;; noqa
         tr-tensor-device/raw
         define-unary/raw
         define-binary/raw
         define-scalar/raw)

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
;;
;; Swallows are silent BY DESIGN (a finalizer has nowhere to report) but
;; not invisible: each one bumps a counter, so leak-by-swallow is
;; observable (finalizer-failures, on the public surface) without
;; reintroducing the cascade. The increment runs under the ledger's
;; atomic guard — the handler itself must never raise.
(define finalizer-failure-count (box 0))

(define (finalizer-failures)
  (unbox finalizer-failure-count))

(define ((guard-finalizer release) t)
  (with-handlers ([(lambda (_) #t)
                   (lambda (_e)
                     (call-with-ledger
                      (lambda ()
                        (set-box! finalizer-failure-count
                                  (add1 (unbox finalizer-failure-count))))))])
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
  ;; Snapshot the entries inside the critical section (one list build),
  ;; fold outside it — keeping the atomic stretch short like account!'s.
  ;; A GC dropping weak keys DURING the snapshot is tolerated: CS weak-
  ;; hash iteration skips collected entries rather than invalidating
  ;; (verified empirically — 50 rounds of mid-iteration collection of
  ;; ~95% of keys, zero raises), which is exactly the approximate-gauge
  ;; semantics documented above.
  (define entries (call-with-ledger (lambda () (hash-values allocations))))
  (define totals (make-hash))
  (for ([a (in-list entries)])
    (hash-update! totals (allocation-device a)
                  (lambda (n) (+ n (allocation-nbytes a)))
                  0))
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

;; --- graceful OOM: kind probe + collect-and-retry (#38 leg 1.5) ----------

;; The error-kind probe — third probe hosted here for the same
;; cycle-free reason as the nbytes/device pair; raw/global.rkt
;; re-provides it beside tr-last-error/raw (mirroring raw/device.rkt). 0 generic, 1 out-of-memory — the C side records it
;; together with every message, so it always describes the LAST failure.
(define-torch tr-last-error-kind/raw
  (_fun -> _int)
  #:c-id tr_last_error_kind)

(define (last-error-oom?)
  (= 1 (tr-last-error-kind/raw)))

;; Fourth probe (same cycle-free rationale; raw/device.rkt re-provides):
;; hands the CUDA caching allocator's unused cached blocks back to the
;; driver. No-op success without CUDA (or in a CPU-only build), so the
;; drain below calls it unconditionally.
(define-torch tr-cuda-empty-cache/raw
  (_fun -> _int)
  #:c-id tr_cuda_empty_cache)

;; Collect, then wait (bounded) for the finalizer executor to work
;; through the frees the collection queued. register-finalizer callbacks
;; run ASYNCHRONOUSLY on the runtime's executor thread — the repo's own
;; finalizer tests settle with collect+wait rounds for exactly this
;; reason — so a retry issued immediately after collect-garbage could
;; refail against native memory whose release is still queued. The
;; canary is registered after the dead handles' finalizers; observing it
;; run is a strong (not guaranteed — execution order across objects is
;; unspecified) signal the queue that includes them has drained, and the
;; timeout bounds the wait when the canary survives this collection or
;; the executor is busy. Worst case is a bounded pause on a path that
;; has already failed once.
;;
;; Known window, deliberate: a default-device constructor retried after
;; the drain re-reads the process-global default, so a CONCURRENT
;; set-default-device! landing during the wait can place the retried
;; tensor per the new default. That is the documented semantics of the
;; mutable global (any constructor racing set-default-device! has
;; nondeterministic placement; the retry widens an existing window, it
;; doesn't create one), and snapshot/re-set from in here would clobber
;; the other thread's deliberate change — worse. Programs that need
;; placement invariants under concurrency pass #:device / use
;; with-default-device scoping; catalogued with the rest of the
;; concurrent-default questions in #40.
(define (collect-and-drain!)
  (define drained (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post drained)))
  (collect-garbage)
  (void (sync/timeout 0.5 drained))
  ;; The drained finalizers returned their blocks to the CACHING
  ;; allocator; before the retry re-asks the driver, hand the cache's
  ;; unused blocks back too (best-effort: rc ignored — a failure here
  ;; must not preempt the retry, whose own failure raises properly).
  (void (tr-cuda-empty-cache/raw)))

;; Retry a NULL-returning raw call exactly once after a collect+drain
;; when the C side classified the failure as OOM: finalizing dead handles
;; returns their blocks to the allocator, so the one measured failure mode
;; leg 1's pressure can't fully close — an unlucky allocation striking
;; between majors with dead handles pending — becomes transparent
;; recovery. A second failure falls through to the caller (error.rkt
;; raises the typed exn there). Retry is safe ONLY for calls that are
;; effect-free on failure; ops that draw from the global RNG stream may
;; consume generator state before the failing allocation, so they take
;; tensor-allocator/rng below instead. Exposed as a combinator with
;; injectable probes so the mechanism is unit-testable without provoking
;; real exhaustion.
(define ((oom-retry #:oom? [oom? last-error-oom?]
                    #:collect! [collect! collect-and-drain!])
         raw-fn)
  (lambda args
    (define t (apply raw-fn args))
    (cond
      [t t]
      [(oom?)
       (collect!)
       (apply raw-fn args)]
      [else #f])))

;; Register the GC finalizer and charge the #37 ledger on each fresh
;; handle. NULL (error) results are never charged.
(define ((accounted wrapped) . args)
  (define t (apply wrapped args))
  (when t (account! t))
  t)

;; The one wrap every tensor-returning raw binding carries (hand-written
;; via the op-definer macros below and the explicit #:wrap sites; generated
;; via define-generated.rkt): GC-managed lifetime, the #37 pressure charge,
;; and the OOM collect-and-retry. The retry composes OUTSIDE the allocator
;; wrap: ffi/unsafe/alloc runs the wrapped call in ATOMIC MODE, where the
;; drain's blocking wait is an internal error ("attempt to deschedule the
;; current thread in atomic mode" — caught by the test suite when the
;; retry sat inside). Correctness is unchanged: the allocator registers a
;; finalizer only on a non-NULL result, so a failed first call registers
;; nothing and the retried call's handle is registered exactly once.
(define (tensor-allocator raw-fn)
  (accounted ((oom-retry) ((allocator tr-tensor-free/finalizer) raw-fn))))

;; tensor-allocator minus the retry, for bindings that consume the global
;; RNG stream (randn/rand; dropout via the generated layer's #:rng flag):
;; a blind retry would draw twice and silently break seeded parity. If an
;; RNG op ever needs the retry, generator snapshot/restore is the
;; mechanism, as its own considered change (see the plan).
(define (tensor-allocator/rng raw-fn)
  (accounted ((allocator tr-tensor-free/finalizer) raw-fn)))

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
