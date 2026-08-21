#lang racket/base

;; Safe tensor operations built on the raw layer + wrapper struct.  These are
;; the implementation behind the contracts in ../foreign.rkt.

(require (only-in ffi/vector f32vector->list list->s64vector make-f32vector)
         (only-in "device-type.rkt"
                  cpu-device cuda-device device-index device-type device?)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/device.rkt"
                  tr-cuda-device-count/raw
                  tr-cuda-empty-cache/raw
                  tr-cuda-is-available/raw
                  tr-cuda-memory-stats/raw
                  tr-get-default-device/raw
                  tr-set-default-device/raw
                  tr-tensor-device/raw
                  tr-tensor-to-device/raw)
         (only-in "raw/global.rkt" tr-manual-seed/raw tr-version/raw)
         (only-in "raw/memory.rkt"
                  collect-and-drain!
                  finalizer-failures
                  native-memory-use)
         (only-in "raw/random.rkt" tr-rand/raw tr-randn/raw tr-tensor-uniform!/raw)
         (only-in "raw/tensor.rkt"
                  tr-tensor-copy-data/raw
                  tr-tensor-item/raw
                  tr-tensor-numel/raw
                  tr-tensor-to-dtype/raw)
         (only-in "structs.rkt"
                  handle->repr
                  handle->string
                  tensor-handle
                  tensor-impl-shape
                  wrap-tensor))

(provide torch-version
         cuda-empty-cache!
         cuda-memory-stats
         reclaim-native-memory!
         device->type+index
         finalizer-failures
         native-memory-use
         manual-seed!
         randn
         rand
         uniform!
         item
         to-dtype
         cuda-available?
         cuda-if-available
         cuda-device-count
         set-default-device!
         default-device
         call-with-default-device
         with-default-device
         to-device
         tensor-device
         tensor-numel
         tensor-shape
         tensor->vector
         tensor->list
         tensor->string
         tensor->repr)

(define (torch-version)
  (tr-version/raw))

(define (manual-seed! seed)
  (check-ok (tr-manual-seed/raw seed) 'manual-seed!)
  (void))

(define (randn . dims)
  (wrap-tensor
   (check-handle 'randn (tr-randn/raw (list->s64vector dims) (length dims)))))

;; Uniform draws on [0, 1), torch.rand.
(define (rand . dims)
  (wrap-tensor
   (check-handle 'rand (tr-rand/raw (list->s64vector dims) (length dims)))))

;; In-place fill with uniform draws on [low, high) — torch.Tensor.uniform_,
;; the RNG primitive behind PyTorch's nn.Linear init.
(define (uniform! t low high)
  (check-ok (tr-tensor-uniform!/raw t
                                    (exact->inexact low)
                                    (exact->inexact high))
            'uniform!)
  (void))

;; The value of a one-element tensor as a Racket real (torch.Tensor.item).
(define (item t)
  (define-values (rc v) (tr-tensor-item/raw t))
  (check-ok rc 'item)
  v)

;; Copy converted to 'float32 / 'float64 / 'int64 (torch.Tensor.to).
(define (to-dtype t dtype)
  (wrap-tensor (check-handle 'to-dtype (tr-tensor-to-dtype/raw t dtype))))

;; A device argument is a device struct (device-type.rkt) or a legacy form:
;; 'cpu, 'cuda (ordinal 0), (list 'cuda ordinal). The FFI uses a separate
;; type symbol + index; queries normalize to device structs.
(define (device->type+index dev)
  (cond
    [(device? dev) (values (device-type dev) (device-index dev))]
    [(eq? dev 'cpu) (values 'cpu 0)]
    [(eq? dev 'cuda) (values 'cuda 0)]
    [(pair? dev) (values 'cuda (cadr dev))]
    [else (error 'device "unsupported device: ~e" dev)]))

(define (type+index->device type index)
  (if (eq? type 'cpu) (cpu-device) (cuda-device index)))

;; #t when a CUDA device is present and usable (torch.cuda.is_available). #f
;; means no CUDA *or* a rare driver/init failure; the C side records the latter
;; in tr_last_error, but this predicate doesn't surface it (it stays a boolean).
(define (cuda-available?)
  (= 1 (tr-cuda-is-available/raw)))

;; The pick-the-accelerator idiom every example re-defines locally as
;; pick-device, now offered by the library: the GPU when one is usable,
;; the CPU otherwise. (The literate examples still teach their own
;; pick-device — folding them over to this is deliberate follow-up work,
;; since their prose walks through the idiom.)
(define (cuda-if-available)
  (if (cuda-available?) (cuda-device) (cpu-device)))

;; Number of visible CUDA devices, 0 when CUDA is unavailable (see
;; cuda-available? re: a driver-failure 0).
(define (cuda-device-count)
  (tr-cuda-device-count/raw))

;; The CUDA caching allocator's gauges for one device, in bytes — an
;; alist of allocated (live blocks), reserved (live + cached), and
;; peak-allocated. Complements native-memory-use: the ledger reports
;; what rktorch's handles hold; this reports what the allocator holds.
;; Errors without CUDA (torch.cuda.memory_allocated & co).
(define (cuda-memory-stats [dev (cuda-device)])
  (define-values (type index) (device->type+index dev))
  (unless (eq? type 'cuda)
    (error 'cuda-memory-stats "expected a CUDA device, given: ~e" dev))
  (define-values (rc allocated reserved peak)
    (tr-cuda-memory-stats/raw index))
  (check-ok rc 'cuda-memory-stats)
  (list (cons 'allocated allocated)
        (cons 'reserved reserved)
        (cons 'peak-allocated peak)))

;; Hand the caching allocator's unused cached blocks back to the driver
;; (torch.cuda.empty_cache). No-op without CUDA; the OOM retry already
;; runs this automatically before retrying. NOTE: dead-but-unfinalized
;; tensors' blocks are not yet IN the cache — for the full
;; release-everything-now sequence use reclaim-native-memory!.
(define (cuda-empty-cache!)
  (check-ok (tr-cuda-empty-cache/raw) 'cuda-empty-cache!)
  (void))

;; The release-it-all-now sequence, in the only order that works:
;; collect (queues dead handles' finalizers), DRAIN the asynchronous
;; finalizer executor (their frees return blocks to the caching
;; allocator), then hand the cache's unused blocks to the driver. The
;; OOM retry runs one bounded round of this internally; the PUBLIC
;; sequence settles instead — repeat while the ledger keeps shrinking
;; (bounded rounds), so a queue longer than one drain window still
;; empties — and the final cache release is CHECKED (an explicit
;; reclaim should surface a driver failure, unlike the retry's
;; best-effort pass).
(define (reclaim-native-memory!)
  (let loop ([prev (ledger-total)] [rounds 4])
    (define drained? (collect-and-drain!))
    (define now (ledger-total))
    ;; go again while the ledger shrinks OR the drain went UNOBSERVED
    ;; (a busy executor can make zero progress in one window without
    ;; being done — no-progress only terminates once the canary was
    ;; actually seen draining), always within the round bound.
    (when (and (> rounds 1)
               (or (< now prev) (not drained?)))
      (loop now (sub1 rounds))))
  (cuda-empty-cache!))

;; total handle-attributed bytes across devices (settling probe above)
(define (ledger-total)
  (for/sum ([entry (in-list (native-memory-use))])
    (cdr entry)))

;; Set the device new tensors (randn/zeros/...) are created on. Errors if CUDA
;; is requested but unavailable or the ordinal is out of range.
(define (set-default-device! dev)
  (define-values (type index) (device->type+index dev))
  (check-ok (tr-set-default-device/raw type index) 'set-default-device!)
  (void))

(define (default-device)
  (define-values (rc type index) (tr-get-default-device/raw))
  (check-ok rc 'default-device)
  (type+index->device type index))

;; Run `thunk` with the process default device set to `dev`, restoring the prior
;; default on the way out (even on escape) — the device analogue of
;; call-with-no-grad. Use this rather than a hand-rolled dynamic-wind so a
;; transient device switch can't leak onto later tensors.
(define (call-with-default-device dev thunk)
  (define saved (default-device))
  (dynamic-wind (lambda () (set-default-device! dev))
                thunk
                (lambda () (set-default-device! saved))))

(define-syntax-rule (with-default-device dev body ...)
  (call-with-default-device dev (lambda () body ...)))

;; Copy a tensor onto `dev` (torch.Tensor.to(device)).
(define (to-device t dev)
  (define-values (type index) (device->type+index dev))
  (wrap-tensor
   (check-handle 'to-device (tr-tensor-to-device/raw t type index))))

(define (tensor-device t)
  (define-values (rc type index) (tr-tensor-device/raw t))
  (check-ok rc 'tensor-device)
  (type+index->device type index))

(define (tensor-numel t)
  (define-values (rc n) (tr-tensor-numel/raw t))
  (check-ok rc 'tensor-numel)
  n)

;; Cached at wrap time, so no C round-trip.
(define (tensor-shape t)
  (tensor-impl-shape t))

(define (tensor->vector t)
  (define numel (tensor-numel t))
  (define out (make-f32vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data/raw t numel out))
  (check-ok rc 'tensor->vector)
  out)

(define (tensor->list t)
  (f32vector->list (tensor->vector t)))

;; ATen's C++ `operator<<` text (the libtorch-native printer).
(define (tensor->string t)
  (handle->string (tensor-handle t)))

;; The PyTorch `repr` text -- identical to what the REPL prints (see structs.rkt).
(define (tensor->repr t)
  (handle->repr (tensor-handle t) (tensor-shape t)))
