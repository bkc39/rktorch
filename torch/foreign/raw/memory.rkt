#lang racket/base

;; Tensor lifetime + native-memory substrate: the explicit/finalizer free
;; split, the phantom-bytes pressure ledger, the OOM collect-and-retry,
;; and the tensor-allocator wrap every tensor-returning binding carries.

(require (for-syntax racket/base
                     ;; whole-module: syntax-parse patterns reference many
                     ;; of its bindings
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
         collect-and-drain!
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

(define-torch tr-tensor-free/unwrapped
  (_fun _Tensor -> _void)
  #:c-id tr_tensor_free)

;; Explicit-free path, raising: the (deallocator) wrap CANCELS the pending
;; GC finalizer — without it the finalizer later frees the dead handle
;; and raises inside finalization.
(define tr-tensor-free/checked
  (let ([release ((deallocator) tr-tensor-free/unwrapped)])
    (lambda (t)
      (unaccount! t)
      (release t))))

;; Swallow EVERYTHING raised out of a finalizer (every value, not just
;; exn:fail): a raise escaping GC finalization re-enters the error
;; machinery, which allocates, re-triggers GC, and loops — the #38
;; cascade. Runs on the runtime's finalizer thread, so swallowing a break
;; loses nothing; each swallow bumps finalizer-failures — a finalizer has
;; nowhere to report, and leaking one handle beats a cascade.
;; (A C++ throw during release dies inside libtorch's noexcept frames
;; before any handler — this guard never even runs for that class.)
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
;; Racket's GC sees a tensor as a tiny wrapper, so each allocation charges
;; a phantom-bytes object sized to its nbytes — collection scheduling then
;; scales with native usage. The ledger is a weak-keyed side table (self-
;; healing: a missed unaccount! collects with the handle); per-device
;; totals are FOLDED ON QUERY, never incremental, so hot-path ops are
;; single hash ops with no read-modify-write across yield points. Reports
;; handle-attributed bytes, NOT device usage: views over-count shared
;; storage, and ATen-internal buffers never cross this boundary.

(struct allocation (phantom nbytes device))

(define allocations (make-weak-hasheq))

;; ATOMIC MODE, not a semaphore: finalizers already run in atomic mode,
;; where blocking on a semaphore is an internal error — call-as-atomic is
;; reentrant there and never blocks.
(define (call-with-ledger thunk)
  (call-as-atomic thunk))

;; Cycle-free canonical home for these probes (the accounting needs them;
;; raw/device.rkt requires this module and re-provides them).
(define-torch tr-tensor-nbytes/raw
  (_fun _Tensor (out : (_ptr o _int64)) -> (rc : _int) -> (values rc out))
  #:c-id tr_tensor_nbytes)

;; Mirrors the tr_device_type C enum (device.h).
(define _tr-device-type
  (_enum '(cpu = 0 cuda = 1)))

(define-torch tr-tensor-device/raw
  (_fun _Tensor
        (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_tensor_device)

;; Best-effort: accounting must never fail an allocation that succeeded
;; (make-phantom-bytes itself can raise under pressure). exn:fail? only,
;; NOT a total catch — on this user-thread path breaks must propagate.
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

;; Releases the pressure charge NOW rather than waiting for the phantom's
;; own collection; guarded like account! — bookkeeping must never turn a
;; free into a failure.
(define (unaccount! t)
  (with-handlers ([exn:fail? void])
    (call-with-ledger
     (lambda ()
       (define a (hash-ref allocations t #f))
       (when a
         (set-phantom-bytes! (allocation-phantom a) 0)
         (hash-remove! allocations t))))))

;; Approximate by design: a query racing heavy allocation sees a
;; snapshot, which is correct for a gauge.
(define (native-memory-use)
  ;; A GC dropping weak keys during the snapshot is tolerated: CS weak-
  ;; hash iteration skips collected entries rather than invalidating.
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

;; FINALIZER-context entry point only (explicit frees use /checked).
;; Built on the UNWRAPPED binding: routing through the (deallocator) wrap
;; would cancel the very registration this finalizer is running from.
(define tr-tensor-free/finalizer
  (guard-finalizer
   (lambda (t)
     (unaccount! t)
     (tr-tensor-free/unwrapped t))))

;; --- graceful OOM: kind probe + collect-and-retry ------------------------

;; 0 generic, 1 out-of-memory; the C side records it with every message,
;; so it always describes the LAST failure. Re-provided by raw/global.rkt.
(define-torch tr-last-error-kind/raw
  (_fun -> _int)
  #:c-id tr_last_error_kind)

(define (last-error-oom?)
  (= 1 (tr-last-error-kind/raw)))

;; Hands the CUDA caching allocator's unused blocks back to the driver.
;; No-op success without CUDA, so the drain calls it unconditionally.
(define-torch tr-cuda-empty-cache/raw
  (_fun -> _int)
  #:c-id tr_cuda_empty_cache)

;; Finalizers run ASYNCHRONOUSLY on the runtime's executor thread, so a
;; retry issued straight after collect-garbage could refail against
;; memory whose release is still queued: register a canary after the dead
;; handles and wait (bounded) for it. Observing it drain is a strong, not
;; guaranteed, signal (finalization order is unspecified). Returns #t
;; when the canary was observed, #f on timeout — settling callers read #f
;; as "executor still busy, go again".
;;
;; Known window (#40): a default-device constructor retried after the
;; wait re-reads the mutable global, so a concurrent set-default-device!
;; can re-place the retried tensor — the global's documented semantics;
;; callers needing placement invariants pass #:device.
(define (collect-and-drain!)
  (define drained (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post drained)))
  (collect-garbage)
  (define observed (sync/timeout 0.5 drained))
  ;; Drained frees land in the CACHING allocator; hand its unused blocks
  ;; back to the driver too. Best-effort: a failure here must not preempt
  ;; the retry, whose own failure raises properly.
  (void (tr-cuda-empty-cache/raw))
  (and observed #t))

;; Retry a NULL-returning raw call exactly once after a collect+drain
;; when the C side classified the failure as OOM; a second failure falls
;; through to the caller (error.rkt raises the typed exn). Safe ONLY for
;; calls that are effect-free on failure — ops drawing from the global
;; RNG stream take tensor-allocator/rng instead.
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

(define ((accounted wrapped) . args)
  (define t (apply wrapped args))
  (when t (account! t))
  t)

;; The one wrap every tensor-returning raw binding carries: GC-managed
;; lifetime, pressure charge, OOM collect-and-retry. The retry composes
;; OUTSIDE the allocator wrap — ffi/unsafe/alloc runs the wrapped call in
;; ATOMIC MODE, where the drain's blocking wait is an internal error. The
;; allocator registers a finalizer only on a non-NULL result, so the
;; retried handle is registered exactly once.
(define (tensor-allocator raw-fn)
  (accounted ((oom-retry) ((allocator tr-tensor-free/finalizer) raw-fn))))

;; tensor-allocator minus the retry, for bindings that consume the global
;; RNG stream: a blind retry would draw twice and break seeded parity.
(define (tensor-allocator/rng raw-fn)
  (accounted ((allocator tr-tensor-free/finalizer) raw-fn)))

;; --- op-definer macros ---------------------------------------------------

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
