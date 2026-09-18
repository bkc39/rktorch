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
                            allocator on CPU, CUDA, or MPS
```

Two handles may share one storage (views). Freeing a handle drops one
reference; the buffer is released when the last reference goes. On
CUDA and MPS the caching allocators usually just return the block to
their pools.

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
an incremented counter (`finalizer-failures`) plus, up to a bound of
eight, the exception's message — read both with
`(finalizer-diagnostics)`, which also reports total runs and live
ledger entries, and is dumped to stderr at exit under
`RKTORCH_MEM_TRACE` (any value). The handler that records this carries a total
guard of its own, because it is the direct body of that catch and an
escape from there is a process death, not a raise. One
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

### Pressure-driven collection

The policy lives in `torch/foreign/raw/pressure.rkt`, under the ledger:
`raw/memory.rkt` tells it of every accounting and release
(`note-accounted!`, `note-unaccounted!`, inside the ledger's atomic
section) and asks it to act. It binds no C itself. Its two CUDA readings,
capacity and the allocator's allocated bytes, are bound in
`raw/device.rkt` with the rest of `device.cpp` and handed down by
`install-cuda-queries!` when that module is instantiated, because
`raw/device.rkt` sits above the ledger and requiring it from below would
be a cycle. One collection runs at a time (`call-as-the-collector`): a
second thread that finds a trigger due while the first is collecting
skips. The claim names its thread, so one killed mid-collection, whose
`dynamic-wind` exit never runs, stops holding it.

Phantom pressure keeps Racket's generational collections running, and
on a GPU training loop they free most of a step's intermediates within
the step. Two things escape them (#145, measured with
`scripts/bench-memory-pressure.rkt`):

- a slow residue of handles promoted past the young generations, about
  12 MiB per step on a 15 GB working set;
- the fatal one: when Racket runs a *major* collection in the middle of
  a forward pass, every intermediate alive at that moment is promoted
  to the oldest generation, and Racket schedules its next major for
  when memory use has doubled from there, which on a card that is more
  than half full is never. That step's storage, a whole working set of
  it, stays behind and the next step's `backward!` fails.

The ledger closes the gap itself, first at the trough and then with a
backstop at the peak.

**The trough.** `backward!` ends by calling `collect-at-trough!`. At that
moment the graph has been released, the forward's intermediates are dead
and almost nothing is live, so a full collection reclaims the most,
promotes the least, and leaves Racket's doubling rule a small baseline
to double from. The dead intermediates have aged past the nursery by
then (a minor collection there was tried and reclaimed next to nothing),
so the collection is a full one, drained as below. It runs when the
ledger exceeds its *floor*, its size after the previous trough
collection and zero before the first, by a margin: the floor itself,
kept between 256 MiB and 1 GiB (`native-collect-margin` overrides it). A time budget spaces them: after a collection that took t, the
next waits t / `native-collect-budget`, 1/20 by default. Both are
parameters on the facade, with `native-memory-limit` and
`native-memory-fraction`, so a script can tune the policy for its own
machine by wrapping its loop in one `parameterize`. On the 35.7M-parameter
DDPM UNet at batch 224 that is a collection every three or four steps,
a peak flat at 15.0 GB against 13.9 GB for a hand collection on every
step, and 0.45 s per step against 0.42 s. Collecting at every trough
instead costs 30%.

**The no-grad trough.** A sampling or evaluation loop never reaches
`backward!`. Its trough is the return of an *outermost* layer call with
gradients off: `call-forward` in `torch/nn/layer.rkt` marks the
continuation for the extent of a call, so a nested call sees the mark
and only the outermost return reaches `collect-at-forward-trough!`,
which checks the grad mode and calls `collect-at-trough! #:young? #t`.
The mark unwinds with the continuation, so an exception leaves no state
behind. This garbage is young, so a minor collection goes first and a
full one takes what survives (handles that lived through a minor
collection during a long forward); the two stages split the budget.
With gradients on, the return of a forward is the peak, the graph
holding every intermediate, and nothing is done. On the #138 sampler
(1000 steps, a doubled batch of 60): window peak 14 GB down to 5.6 GB,
reserved 14.6 GB down to 6.0 GB, 70.7 s against 67.4 s.

**The backstop.** A collection at the peak is the wrong moment, since it
promotes that step's live intermediates and so sets up the next one,
but it is the only moment available to a loop with no `backward!`, and
the last defence when a trough collection is not yet due:

- Each device has one `account` record: live bytes, bytes accounted
  since the last check and since the last allocator sample, that sample,
  the backstop's current interval, the cached capacity mark and the
  trough floor. `account!` updates it through `note-accounted!`, so a
  check is one table lookup and field reads. The counters are a `stats`
  record, the two next-collection times and the thread inside a
  collection a `schedule` record, and the two CUDA queries handed down
  from `raw/device.rkt` a `queries` record. Every field is written
  inside `call-with-ledger`.
- `accounted`, which runs outside the allocator's atomic wrap, checks
  after each accounting: accounted-since past the current interval and
  live bytes above the device's high-water mark means
  `collect-and-wait!`, the drain half of `collect-and-drain!` without
  the cache emptying, which is the OOM retry's business.
- Live bytes are the larger of the ledger's counter and the caching
  allocator's `allocated`, sampled once per 1/32 of the mark in
  accounted bytes (`allocator-reading`, a parameter so tests can fake
  it). `raw/device.rkt` installs both readings with
  `install-device-queries!`; each takes a device and dispatches on its
  type, so CUDA and MPS share one path and the CPU answers `#f`. Handles the young collections free mid-step leave their storage
  with the autograd graph, where only the allocator can see it.
- `collect-and-wait!` must really drain. The canary shows the finalizer
  thread has started on the batch, not finished it: finalization order
  is unspecified, and that thread runs only while the main one yields,
  which a loop of FFI calls does not do. So after the canary it yields
  until `finalizer-run-count` has stood still for three turns, within
  two seconds. Without this a collection found 13 GB of garbage and
  the next `backward!` still failed, the frees not yet made.
- The mark is `native-memory-fraction` of the device's capacity, 80% by
  default: `tr_cuda_mem_get_info`'s total on CUDA and
  `tr_mps_memory_info`'s recommended maximum on MPS. The capacity is
  cached once known, and a failed query on a device that should have one
  is retried after a second, so one early failure cannot switch the
  backstop off. The fraction is applied at each check, not folded into
  the cache, so a program may change it mid-run; `native-memory-limit` overrides it for every
  device and is how the CPU tests exercise the path. No capacity and no
  limit means the check is off.
- Hysteresis: the interval starts at an eighth of the mark; a
  collection that reclaims under 5% of the mark doubles it (capped at
  twice the mark), one that reclaims more resets it, and one whose
  drain ran out of time leaves it alone. A stalled drain also leaves the
  bytes-since-check counters and the trough floors where they were: it
  measured nothing, so it earns no credit, and the next allocation looks
  again. Looking is cheap; the mark and the trough margin still guard the
  collection itself. A working set that
  sits above the mark is therefore collected at a decaying rate instead
  of on every allocation.
- Never from atomic mode (`in-atomic-mode?` guards it), and the RNG
  wrap gets the check too: it runs after the draw, so it cannot
  double-draw.
- `finalizer-diagnostics` reports `trough-collections`, `trough-minors`,
  `pressure-collections` (the backstop) and `pressure-reclaimed` (bytes,
  all kinds).

### In-place moves

`to` on a layer moves each parameter and buffer through
`tr_tensor_to_`, which rebinds the storage under the existing handle
(`Tensor::set_data`, the mechanism behind `torch.nn.Module.to`). The
handle's ledger entry recorded the old device and byte count, so the
Racket side re-accounts it (`reaccount!`: unaccount, then account) as
the same weak key — one entry per handle before and after, now in the
destination's bucket. libtorch releases the old storage at `set_data`
time when nothing else references it. A wrapper that aliased the
tensor before the move keeps its stale charge until it dies, the
documented over-count. The functional `to` needs no special
treatment: its result is a fresh allocation charged like any other,
and when nothing would change it returns the source object rather
than a second wrapper over the same storage.

## Thread safety

- The native side is safe by construction: `at::Tensor` refcounts are
  atomic, each handle owns an independent reference, and handles
  sharing storage may be freed in either order from any thread.
- The ledger is serialized with `call-as-atomic`, not a lock: the
  finalizer side already runs in atomic mode, where taking a semaphore
  would deadlock. Per-device live totals are counters updated in the
  same atomic section as the entry, so the pressure check costs one
  hash lookup per allocation; the entry-by-entry fold survives as
  `native-memory-use/fold`, the cross-check the tests run.
- The finalizer failure count is likewise incremented atomically in
  the guarded finalizer context.

## Allocation failure

When an allocation fails and classifies as out-of-memory, the wrapper
collects, drains pending finalizers (a bounded, best-effort drain),
and retries the call exactly once; a second failure
raises the typed `exn:fail:rktorch:oom`. Classification is by exception
type (`c10::OutOfMemoryError`, `std::bad_alloc`) plus message shape for
the allocators that only throw plain `c10::Error`: the CPU allocator's
"DefaultCPUAllocator" and the MPS allocator's two refusals — "MPS
backend out of memory" (high-water mark) and "Invalid buffer size:"
(Metal's per-buffer cap) — so an oversized request gets the typed OOM
on every backend. Ops that draw from the global
RNG stream use `tensor-allocator/rng` — the same wrap minus the retry,
because a retried draw would advance the generator stream and break
seeded reproducibility.

## Observability and control

- `native-memory-use` — the ledger fold, per device. A
  handle-attributed estimate (views double-count; ATen-internal
  allocations absent).
- `cuda-memory-stats` — the CUDA caching allocator's own
  allocated/reserved/peak numbers for this process.
- `cuda-empty-cache!` / `mps-empty-cache!` — return
  reserved-but-unused blocks to the driver; each is a no-op success
  when its backend is absent.
- `reclaim-native-memory!` — collect, drain, repeat (a bounded number
  of rounds) until the ledger stops shrinking, then empty the CUDA and
  MPS caches. For epoch boundaries and script exits.
- `tensor-free!` — deterministic release of one handle's reference
  (unsafe submodule); the buffer goes when the last sharing handle
  does.
- `finalizer-failures` — the swallowed-failure counter; nonzero means
  frees failed silently and the process should be treated as wounded.
