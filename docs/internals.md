# rktorch internals: native memory management

rktorch tensors are libtorch tensors. Racket sees each one as a small
wrapper struct; the buffer — megabytes of float32 on the host or the
GPU — lives on the native side, invisible to Racket's GC. The job of
this layer is to make the GC see those bytes, so that ordinary
collection keeps native memory honest on every device.

## Indirection

```
  (tensor ...)              Racket struct: prop:cpointer + cached shape
      |
  cpointer 'Tensor          tagged handle; tag flips to 'Tensor-freed
      |                     after any free attempt
  tr_tensor*                C heap object (one per handle)
      |
  at::Tensor                libtorch value: refcounted TensorImpl
      |
  Storage -> DataPtr        the buffer, owned by libtorch's (caching)
                            allocator on CPU or CUDA
```

Two handles may share one storage (views). Freeing a handle drops one
reference; the buffer is released when the last reference goes. On
CUDA the caching allocator usually just returns the block to its pool.

## Tensor lifetime

Every tensor-returning binding carries the `tensor-allocator` wrap
from `torch/foreign/raw/memory.rkt` (for generated ops, the
`define-generated-op` expansion inserts it). The wrap does two things: registers a GC finalizer for
the handle, and charges the pressure ledger below. A bare
`(allocator ...)` wrap is never written — it would skip the ledger.

Death has two paths:

| | explicit (`tensor-free!`) | GC finalizer |
|---|---|---|
| entry | `tr-tensor-free/checked` | `tr-tensor-free/finalizer` |
| error behavior | raises | swallowed + counted |
| finalizer | cancelled | is the finalizer |
| when | deterministic release | the default |

The explicit path drops this handle's reference *now*, cancels the pending
finalizer, and flips the cpointer tag to `'Tensor-freed` — even when
the free raises, since an attempted free consumes the finalizer
backstop and a live-looking tag would invite use-after-free.

The finalizer path is wrapped in `swallow-and-count-failure`, whose
catch is total: Racket finalizers run in **atomic mode**, where
raising or blocking is not an option, so a failed native free becomes
an incremented counter (`finalizer-failures`) and nothing else. One
outcome bypasses both paths: a C++ throw during storage release
unwinds through libtorch's own noexcept frames to `std::terminate`
before any handler can run — `finalizer_death_test.cpp` pins this.

## Phantom-bytes accounting

`make-phantom-bytes` is Racket-CS's way to charge native allocations
to the GC. `tensor-allocator` stores one phantom of the tensor's
`nbytes` in a ledger keyed weakly by the handle:

- weak keys: the entry — and its charged pressure — vanishes with the
  handle; no lifetime coupling beyond the allocator wrap itself.
- `nbytes` is the *view's* extent: views over-charge shared storage,
  a narrow view under-charges its large storage. Over-counting is safe
  for pressure; the approximation is deliberate.
- a failed size probe charges 0 — accounting is best-effort and never
  a new failure mode on the allocation path.

**GPU bytes charge 1:1 with host bytes into the same pool.** This is
the point of the design: pressure scales with total native footprint,
so the ordinary GC cycle collects dead CUDA tensors as gracefully as
host ones — user code never calls the collector by hand.

## Thread safety

- The native side is safe by construction: `at::Tensor` refcounts are
  atomic, each handle owns an independent reference, and handles
  sharing storage may be freed in either order from any thread.
- The ledger is serialized with `call-as-atomic`, not a lock: the
  finalizer side already runs in atomic mode, where taking a semaphore
  would deadlock. Per-device totals are folded on query
  (`native-memory-use`), not maintained as mutable counters.
- The finalizer failure count is likewise incremented atomically in
  the guarded finalizer context.

## Allocation failure

When an allocation fails and classifies as out-of-memory, the wrapper
collects, drains pending finalizers (a bounded, best-effort drain),
and retries the call exactly once; a second failure
raises the typed `exn:fail:rktorch:oom`. Ops that draw from the global
RNG stream use `tensor-allocator/rng` — the same wrap minus the retry,
because a retried draw would advance the generator stream and break
seeded reproducibility.

## Observability and control

- `native-memory-use` — the ledger fold, per device. A
  handle-attributed estimate (views double-count; ATen-internal
  allocations absent).
- `cuda-memory-stats` — the CUDA caching allocator's own
  allocated/reserved/peak numbers for this process.
- `cuda-empty-cache!` — return reserved-but-unused blocks to the
  driver.
- `reclaim-native-memory!` — collect, drain, repeat (a bounded number
  of rounds) until the ledger stops shrinking, then empty the CUDA
  cache. For epoch boundaries and script exits.
- `tensor-free!` — deterministic release of one handle's reference
  (unsafe submodule); the buffer goes when the last sharing handle
  does.
- `finalizer-failures` — the swallowed-failure counter; nonzero means
  frees failed silently and the process should be treated as wounded.
