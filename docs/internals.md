# rktorch internals: native memory management

rktorch tensors are libtorch tensors. Racket sees each one as a small
wrapper struct; the actual buffer — megabytes of float32 on the host or
the GPU — lives on the native side, invisible to Racket's GC. Everything
in this document exists because of that asymmetry and the specific ways
it failed before each mechanism landed (the #37/#38 investigation).

The layering, top to bottom:

```
  (tensor ...)              Racket struct: prop:cpointer + cached shape
      |
  cpointer 'Tensor          tagged handle; tag flips to 'Tensor-freed
      |                     after any free ATTEMPT
  tr_tensor*                C heap object (one per handle)
      |
  at::Tensor                libtorch value: refcounted TensorImpl
      |
  Storage -> DataPtr        the actual buffer, owned by libtorch's
                            (caching) allocator on CPU or CUDA
```

Two handles may share one storage (views); freeing a handle drops one
reference, and the buffer is released only when the last reference
goes. Frees are cheap on the CUDA path: the caching allocator usually
just returns the block to its pool without a driver call.

## The FFI boundary

Every C entry point is `extern "C"` and follows one contract: return a
`tr_tensor*` (NULL on error) or an int status (0 success, nonzero
error), with the failure recorded in `tr_last_error` — a thread-local
string — and `tr_last_error_kind` (0 generic, 1 out-of-memory), always
set as a pair. The Racket side checks the status/NULL, then reads both.

The exception-to-status translation lives in one place,
`cpp/src/torchrkt/detail/op_call.hpp`, as boundary-shape helpers
(`alloc_result`, `status_call`, `copy_data_call`) plus two sanctioned
deviations: the value-returning CUDA probes (which return a benign 0
and record the failure — callers disambiguate via `tr_last_error`), and
the finalizer free described below.

Error recording is allocation-free end to end on its fallback path.
`record_failure` classifies the exception first, then *attempts* the
rich message; if building that string itself throws (host exhaustion),
`set_error_fallback` records the pre-classified kind without
allocating. This matters because the recorder runs inside `noexcept`
boundary frames: an allocation failure while reporting an allocation
failure must not become `std::terminate`.

## Tensor lifetime

A tensor is born through one choke point. Every tensor-returning raw
binding carries the `tensor-allocator` wrap from
`torch/foreign/raw/memory.rkt` (hand-written bindings say it
explicitly; the codegen emitter emits it for generated ops, and the
codegen-drift CI job keeps that honest). The wrap composes three
things: `ffi/unsafe/alloc`'s `allocator` (registers a GC finalizer for
the handle), the pressure ledger charge (next section), and the OOM
retry (section after). A bare `(allocator ...)` wrap is never written
by hand — it would skip the ledger.

Death has two paths, split on purpose:

| | explicit (`tensor-free!`) | GC finalizer |
|---|---|---|
| entry | `tr-tensor-free/checked` | `tr-tensor-free/finalizer` |
| error behavior | raises | swallowed + counted |
| finalizer | cancelled (`deallocator` wrap) | is the finalizer |
| when | deterministic release in scripts/loops | the default |

The explicit path exists so long-running code can release a buffer
*now* rather than at the GC's leisure; it cancels the pending finalizer
(the `(deallocator)` wrap does this pairing) and flips the cpointer tag
to `'Tensor-freed`. The tag flips even when the free raises: once a
free has been attempted the finalizer backstop is consumed, and a
live-looking tag would invite use-after-free.

The finalizer path is wrapped in `swallow-and-count-failure`, whose
catch is total — every value, not just `exn:fail?`. This is the fix for
the original crash class: a CUDA context that has taken a fault turns
the next free into an error *inside a GC finalizer*, and Racket
finalizers run in **atomic mode** — an error handler that itself
touches the GC or blocks loops the runtime into the "invalid memory
reference" cascade that ended as a host OOM kill. Swallowing at the
single choke point turns that cascade into one incremented counter,
readable as `finalizer-failures`.

One class of failure cannot be caught at any layer, and we pinned it
rather than pretending otherwise: a C++ throw during storage release
unwinds through libtorch's own implicitly-`noexcept` frames
(`~TensorImpl`/`~StorageImpl`/`~DataPtr`) and reaches `std::terminate`
before any handler of ours — two catch-based revisions of
`tr_tensor_free` both still aborted. So `tr_tensor_free` is a bare
`delete` with no catch, and `finalizer_death_test.cpp` EXPECT_DEATHs
the behavior so a libtorch upgrade that ever makes the path catchable
flips the test loudly.

## GC pressure: phantom bytes

The measured problem: churning 2 GB of immediately-dropped 4 MB
tensors left **all 2 GB resident** (264 → 2267 MB RSS) — Racket's GC
saw only tiny wrapper structs, felt no pressure, ran no majors, so no
finalizers ran and no native buffer was freed until program exit. At
training scale the same lag parks dead intermediates on the GPU
between incidental collections; on a busy card, one unlucky allocation
then fails.

The fix charges native bytes to the GC using the Racket-CS-supported
mechanism for exactly this, `make-phantom-bytes`. Inside
`tensor-allocator`, each new handle gets a phantom-bytes object of the
tensor's `nbytes` stored in a ledger keyed weakly by the handle:

- weak keys mean the entry — and its charged pressure — vanishes with
  the handle; no lifetime coupling beyond the allocator wrap itself.
- `nbytes` is the *view's* extent (numel × itemsize), not storage
  bytes: two views over one storage double-charge, a narrow view over
  a large storage under-charges. Over-counting is safe for pressure;
  the approximation is the documented trade.
- a failed `nbytes` probe charges 0 and proceeds — accounting is
  best-effort and must never become a new failure mode on the
  allocation path.
- the ledger is folded per-device on query (`native-memory-use`), not
  maintained as running totals; ledger mutation happens in atomic mode
  (`call-as-atomic`) because the *finalizer side* of the ledger runs in
  atomic mode, where a semaphore would deadlock.

Measured on landing: the same 2 GB churn retains 1021 MB high-water
instead of 2267 MB (pressure triggers majors mid-loop, majors run
finalizers, finalizers free buffers); per-op overhead ~1.2%; a
training run squeezed by a 22 GiB balloon that formerly OOMed at step
4 completes.

## Failing gracefully: typed OOM + collect-and-retry

When an allocation does fail, two mechanisms turn it from a crash into
either transparent recovery or one catchable error.

**Classification.** The C boundary distinguishes OOM from everything
else, per backend (probed on libtorch 2.9):

- CUDA/MPS exhaustion throws the typed `c10::OutOfMemoryError` — a
  `dynamic_cast` catches it.
- CPU exhaustion arrives as a *plain* `c10::Error` from
  `alloc_cpu.cpp`'s enforce — classified by matching
  `DefaultCPUAllocator` in the message (and `std::bad_alloc` is OOM
  too). CPU OOM is portably provokable with one absurd request, so
  this classification has a real gtest; the CUDA path can't have one.
- gradual host exhaustion under Linux overcommit arrives as the OOM
  killer (SIGKILL), not an exception — phantom-bytes pressure is the
  defense there, not this leg.

The kind reaches Racket via `tr_last_error_kind` and raises as
`exn:fail:rktorch:oom`, a struct subtype of `exn:fail` — callers catch
OOM **by type**, never by regexing messages.

**The retry.** On NULL + OOM-kind, the wrapper runs one recovery round
and retries the raw call exactly once; a second failure raises the
typed exn. The sequence:

```
  raw call ──NULL──> kind probe ──oom──> collect-and-drain!
      |                  |                   |
   handle             generic             collect-garbage
      |                  |                   + drain finalizers
   return             raise                  (canary-bounded)
                                             |
                                        retry once ──NULL──> raise
                                             |                typed oom
                                          handle
```

`collect-and-drain!` is more than `(collect-garbage)`: finalizers run
asynchronously after a major, so it registers a canary finalizer and
drains the finalization executor until the canary fires (bounded), so
the retry actually sees the freed blocks.

Two composition rules here were paid for in bugs:

- the retry composes **outside** the allocator wrap — the wrap's
  interior runs in atomic mode, where a GC-triggering retry would be
  an internal error.
- RNG-drawing ops (`randn`, `rand`, dropout's training path — flagged
  `rng` in the codegen allowlist) take `tensor-allocator/rng`, the same
  wrap *minus the retry*. ATen may consume generator state before the
  failing allocation, so a blind retry would advance the stream and
  silently break seeded parity — the property the whole test suite's
  golden losses stand on. If an RNG op ever needs the retry,
  generator-state snapshot/restore is the mechanism, as its own
  change.

## Observability and explicit control

Two views of memory, deliberately different:

- `native-memory-use` — the handle-attributed ledger fold, per device.
  An estimate: views double-count shared storage, ATen-internal
  allocations are absent. Answers "what is Racket's footprint charged
  as".
- `cuda-memory-stats` — the CUDA caching allocator's own
  allocated/reserved/peak numbers. Answers "what does the card
  actually hold for this process", without contaminating the answer
  with other processes the way `nvidia-smi` board totals do.

Control surface:

- `tensor-free!` — deterministic release of one tensor (unsafe
  submodule; the GC path needs no help in normal code).
- `reclaim-native-memory!` — the full settle loop: collect, drain
  finalizers, repeat until the ledger stops shrinking (bounded
  unconditional rounds), then `cuda-empty-cache!`. For epoch
  boundaries and script exits.
- `cuda-empty-cache!` — return reserved-but-unused blocks to the
  driver, so a finished process isn't squatting on reservation
  high-water next to other jobs.
- `finalizer-failures` — the swallowed-failure counter; nonzero after
  a run means frees failed silently (a faulted CUDA context, usually)
  and the process should be treated as wounded.

## What is deliberately not here

No C-side callback into Racket mid-allocation (the Racket-side retry
covers the recoverable case without re-entrancy). No per-storage
accounting or custodian scoping. No allocator configuration surface —
`PYTORCH_CUDA_ALLOC_CONF` remains an environment variable. Charged
CUDA bytes weigh 1:1 with host bytes until measurement demands
otherwise.
