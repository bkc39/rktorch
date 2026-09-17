# Pressure-driven collection in the ledger (#145)

Branch `foreign/memory-pressure-145`, from master at 0a2484f. Refines the
plan in the issue's hand-off comment; the mechanism is confirmed below.

## The mechanism, confirmed (CPU, 2026-09-16)

`scripts/bench-gc-policy.rkt` mimics a step: K temporaries of 4 MiB live
together, then dropped as a batch. Ledger bytes over the baseline after each
step, K = 32 (128 MiB per step):

| between steps | ledger after steps 0..7 (MiB) | finalizer runs/step | ms/step |
|---|---|---|---|
| nothing | 128 128 216 128 232 320 408 128 | ~10, then a burst on a major | 12 |
| `(collect-garbage 'minor)` + yield | 128 128 208 128 208 288 368 128 (peaks 640) | ~12, burst on a major | 12 |
| `(collect-garbage)` + yield | 0 0 0 0 0 0 0 0 | 32 | 55 |

Sixteen minor collections run during every step; they release only the
handles allocated after the last minor collection of the step. Everything
else has been promoted, and only a major collection finds it. The residue
between majors is the creep measured on the 3090 Ti. So the trigger must be
a full collection, and the cost question is how to make full collections
cheap and rare, not how to make minor ones effective.

## Design

### 1. Capacity from the shim

`tr_cuda_mem_get_info(int64_t device_index, int64_t* out_free,
int64_t* out_total)` in `c_api/device.h` + `device.cpp`: `cudaMemGetInfo`
under `TORCHRKT_WITH_CUDA_ALLOCATOR`, the same guard and status shape as
`tr_cuda_memory_stats`. MPS: `recommendedMaxWorkingSetSize` only if the
hooks expose it cheaply, else the entry reports "not available" and the
parameter below covers it. Pinned in `c_api_compile_test.c` and a gtest in
`device_test.cpp`. Raw binding in `raw/device.rkt`; promoted as
`cuda-memory-info` beside `cuda-memory-stats` in `ops.rkt` (`promoted.rkt`
is at 497 lines). The high-water mark uses *total*, not *free*: free
includes other tenants and the caching allocator's own reserve, and would
make the trigger chase the neighbour's jobs.

### 2. Running per-device totals in the ledger

`native-memory-use` folds the whole weak hash on every call, so it cannot
be consulted per op. `account!`/`unaccount!` maintain a per-device total
under the same atomic section (a `hasheq` from device to bytes). The fold
stays as the cross-check in tests; `native-memory-use` reads the counters.
`docs/internals.md` says totals are folded on query; that sentence changes.

### 3. The trigger

In `accounted`, after `account!` (outside the allocator's atomic wrap, on
both `tensor-allocator` and `tensor-allocator/rng`: the draw has already
happened, so the RNG concern does not apply) and after `reaccount!`:

```
live(dev)           > high-water(dev)         ; 80% of capacity, or
                                              ; (native-memory-limit)
accounted-since(dev) >= interval(dev)         ; hysteresis
not (in-atomic-mode?)                         ; never from a finalizer
```

fires `collect+drain` (the canary wait factored out of
`collect-and-drain!`, without the `empty_cache` calls: those are the
expensive cudaFree path and stay on the OOM retry). Hysteresis is the
bytes accounted on that device since the last trigger: `interval` starts
at capacity/8; when a trigger reclaims under 5% of capacity the interval
doubles (cap 2x capacity), when it reclaims more it resets. Without this a
working set legitimately above the mark (batch 256 at 21 GB of 24) would
collect on every step at 0.3 s each.

`native-memory-limit`: a parameter, bytes or `#f`, per device type or
global; the CPU tests set it to a few hundred MiB. When capacity is unknown
and the parameter is `#f` the trigger is off, so small CPU programs never
see it. Diagnostics: `triggers` and `reclaimed-bytes` join
`finalizer-diagnostics`.

### 4. Cost of a full collection

Measure before building anything: `current-gc-milliseconds` and
wall-clock around one trigger at the UNet size, split into collection,
drain (finalizer thread), and `empty_cache`. Only if the drain dominates:
`tr_tensor_free_many(tr_tensor** handles, int64_t n)`; the finalizer then
only pushes onto a queue (atomic, no FFI) and the trigger, the OOM retry
and the exit plumber drain it. Unaccount at drain time, not at queue time,
so the ledger keeps saying what the device holds.

### 5. Not on this branch

Step-level retry in the trainer (#142), allocation-lean schedulers (#144),
`torch/nn/optim.rkt` and `torch/nn/ema.rkt` (in flight on #138).

## Benchmarks

- `scripts/bench-gc-policy.rkt` (CPU, exists): policy x K, ledger peak,
  finalizer runs, ms/step. Before/after the trigger with
  `native-memory-limit` low: peak bounded, overhead per step.
- `scripts/bench-memory-pressure.rkt` (GPU, to write): a stack of
  `Conv2d` + `LayerNorm` + `relu` blocks on random `[N 3 32 32]` batches
  with `adam`, prints `cuda-memory-stats` peak and s/step every ten steps.
  Env: `BATCH`, `STEPS`, `GC_EVERY` (the manual workaround, for the
  comparison row), `LIMIT` (the parameter). Reproduces the creep on master
  at a few GB, so it runs beside the #138 training once that job has
  finished (GPU at 20 GB of 24.5 until roughly 23:30 UTC 2026-09-16).
- The real probe: `~/cifar10-diffusion/train.rkt` at batch 224 with
  `GC_EVERY` unset, on this branch rebased over #138 once it merges.

Acceptance (from the issue): probe flat within 10% of the 14 GB working
set; under 10% per-step overhead at that size; `raco test examples/test/`
timings unchanged; `native-memory-test.rkt`, `oom-error-test.rkt`,
`finalizer-guard-test.rkt`, `native-staging-test.rkt` green; a new CPU
test asserts the ledger stays bounded and the trigger count grows under a
low limit.

## Sequence

1. PR A: capacity entry + running totals + trigger + parameter + CPU test
   + `device.scrbl` and `internals.md`.
2. Measure the collection cost on the GPU probe; PR B for batched frees
   only if the drain is the cost.
3. #142 and #144 stay their own issues.

## Status (2026-09-16)

PR A is built on this branch. The same probe with `POLICY=pressure
LIMIT=256` (the ledger's own trigger, nothing by hand):

| between steps | ledger peak (MiB) | collections in 16 steps | ms/step |
|---|---|---|---|
| nothing | 448 | 0 by hand, 3 incidental majors | 12 |
| `(collect-garbage)` + yield | 0 | 16 | 55 |
| the trigger, 256 MiB limit | 248 | 5 | 35 |

`native-pressure-test.rkt` pins: counters agree with the fold; no limit and
no capacity never fires; a 64 MiB limit bounds a 32 MiB/step churn under
128 MiB; a working set above the mark backs off (at most 5 collections for
120 MiB of live growth).

## GPU results (2026-09-17, 3090 Ti, libtorch 2.9)

`scripts/bench-memory-pressure.rkt`, batch 1024, width 128, depth 8: a
15 GB working set on master's functional adam.

| mode | steps | peak per window (GB) | held after a step (GB) | collections | s/step | gc total |
|---|---|---|---|---|---|---|
| trigger off | 200 | 15.1 rising to 17.2 | 0.15 rising to 2.3 | none | 0.366 | 1.6 s |
| by hand every 10 steps | 200 | 15.0 flat | 0.05 | 20 | 0.386 | 12.8 s |
| trigger, default 80% mark | 700 | 15.1 to 17.2 sawtooth, 19.7 once | 0.2 to 2.3 | 3, reclaiming 37.8 GB | 0.367 | 3.0 s |
| trigger, 15.5 GB mark, before the drain fix | died at 131 | 22 at the failure | 13.3 | 2, reclaiming 0.7 GB | 0.37 | |
| trigger, 15.5 GB mark, with the drain fix | 200 | 16.3 | 0.1 to 1.4 | 1, reclaiming 13.1 GB | 0.372 | 1.4 s |

What the traced failure showed: no major collection for 130 steps, the
young generations keeping up; a major mid-forward in step 131 promoted
13 GB of live intermediates, after which nothing collected them; the
trigger fired in step 132 but measured and resumed before the finalizers
had run. Hence the drain in `collect-and-wait!`, and the allocator reading
in the pressure signal (at the failure the ledger saw 12.6 GB, the
allocator 21.4 GB).

Also added: `cuda-reset-peak-stats!` (the windows above) and
`cuda-allocator-settings!`. Open: whether the library turns on
`expandable_segments` by default; the real diffusion probe on #138's UNet;
the collection-cost split (a drained collection costs about 0.5 s here,
three times in 700 steps, so batched frees have no case yet).
