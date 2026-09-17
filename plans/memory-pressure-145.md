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

## The diffusion probe (2026-09-17)

`~/cifar10-diffusion/train.rkt` on a local merge of this branch with #138
(clean merge, not pushed): the 35.7M-parameter UNet at batch 224, 150 steps,
hand collection disabled (`GC_EVERY=1000000`). The issue's plain loop died
near step 35.

| mark | peak allocated | reserved at end | pressure collections | s/step |
|---|---|---|---|---|
| default, 80% of capacity (19.3 GB) | 21.2 GB | 22.4 GB | not recorded | 0.42 |
| `native-memory-limit` 15 GB | 16.8 GB | 18.2 GB | 26, reclaiming 114 GB | 0.42 |

Both complete with no per-step cost against the plain loop's 0.42 s (these
two rows predate the trough collection below). The
default mark does not meet the issue's "within 10% of the 14 GB working
set": the trigger only acts at accounting points, so the peak is the mark
plus what `backward!` adds on top, 2.9 GB short of the card here. A tighter
mark costs nothing measurable at this size, which argues for a lower
default or a mark relative to the observed working set.

## Collecting at the trough (2026-09-17)

The mid-forward trigger fires at the peak, where a full collection promotes
the step's live intermediates; the 15 GB-mark run above made 26 collections
in 150 steps, each setting up the next. `backward!` now ends with
`collect-at-trough!`; the capacity mark remains as the backstop.

Synthetic bench, batch 1024 (everything off: 0.366 s/step, peak 15.1 GB
and rising):

| trough policy | peak | reserved | trough collections | s/step |
|---|---|---|---|---|
| every trough, minor first then major | 11.4 GB | 15.5 GB | 239 majors in 300 steps | 0.49 |
| 5% time budget, major only | 12.9 GB flat | 14.7 GB | 167 in 400 steps | 0.375 |

The minor stage reclaimed almost nothing (the dead intermediates have aged
out of the nursery by the end of `backward!`), so it was dropped.

The diffusion UNet on the local merge with #138, hand collection disabled:

| run | peak | reserved | trough / backstop | s/step |
|---|---|---|---|---|
| batch 224, 150 steps | 15.0 GB flat from step 10 | 17.1 GB | 46 / 0 | 0.45 (plain 0.42) |
| batch 256, 196 steps (one epoch) | 16.9 GB flat | 19.2 GB | 64 / 0 | 0.49 |
| sampler, 1000 no-grad steps, 60 samples | 14 GB per window | 14.6 GB | 0 / 0 | 67 s total |

Against the issue's acceptance: batch 224 is within 8% of the 13.9 GB that
a hand collection on every step reaches, at 7% overhead; batch 256, which
died in epoch 1 with a 21.1 GB probe peak, runs an epoch flat.

The initial floor and the 5% budget are defaults to review.

## The no-grad trough, and the leak check (2026-09-17)

The return of an outermost layer call with gradients off is now a trough:
minor collection first, a full one for what survives, one shared budget.

| no-grad run | window peak | reserved | backstop / full / minor | time |
|---|---|---|---|---|
| synthetic forward, batch 1024, 300 steps, backstop only | 20.0 GB | 20.5 GB | 193 / 0 / 0 | 0.116 s/step |
| same, full collections only at the trough | 12 to 18 GB | 18.5 GB | 0 / 47 / 0 | 0.120 s/step |
| same, minor first | 12.8 GB (one forward's worth) | 13.4 GB | 0 / 1 / 300 | 0.111 s/step |
| #138 sampler, 1000 steps, before | 14 GB | 14.6 GB | 0 / 0 / 0 | 67.4 s |
| #138 sampler, a budget per stage | 4.7 GB | 5.4 GB | 0 / 96 / 1000 | 76.4 s |
| #138 sampler, one shared budget | 5.6 GB | 6.0 GB | 0 / 47 / 1000 | 70.7 s |

UNet training at batch 224 is unchanged by the layer-call wrapper: 14,988
MiB flat, 0.43 s/step.

Leak check, `scripts/bench-memory-pressure.rkt` sampled along long runs.
Every column a leak would move is flat or oscillating, and dropping the
model returns the ledger, the allocator's allocated and reserved bytes and
the ledger's entry count to zero:

| run | allocated after a step | ledger entries | Racket heap | RSS | after the drop |
|---|---|---|---|---|---|
| no-grad, 3000 steps | 24 MiB, constant | 38, constant | 121 to 130 MiB, oscillating | 798 MiB, constant | 0 / 0 / 0 |
| training, batch 512, 1500 steps | 88 to 129 MiB, no trend | 243 to 336, no trend | 174 to 215 MiB, oscillating | 970 to 971 MiB | 0 / 0 / 0 |

Left open: within one no-grad forward nothing is collected, so the window
peak is one forward's worth of intermediates (a nested trough would tighten
it); `expandable_segments` stays opt-in.

## Epochs without hand collection (2026-09-17, after #138 merged)

Master merged into this branch (clean). The merged arc carries no manual
collection in the repo; the only one was `GC_EVERY` in the out-of-repo
`~/cifar10-diffusion/train.rkt`, now removed there (the original kept beside
it as `train.rkt.with-gc-every.bak`). Batch 224, the 35.7M-parameter UNet:

- 3 epochs plus both sampling passes (60 samples each): complete; epoch
  peaks 16.1 / 17.1 / 16.1 GB; about 98 s per epoch; backstop 0, trough
  collections 288, minors 1706, finalizer failures 0.
- 5 epochs with `reclaim-native-memory!` before measuring at each epoch end
  (a temporary copy of the script; the script itself stays free of it): the
  settled state is identical after every epoch, 1131 MiB in the ledger across
  1980 entries, 1304 MiB allocated, 1239 MiB of Racket heap, epoch peak
  16.3 GB. Nothing accumulates from epoch to epoch.

`scripts/train-asr.rkt` and `scripts/train-gpt.rkt` each call
`reclaim-native-memory!` once after training, to hand cached VRAM back before
generation. That is the function's documented use, not a leak workaround, and
is left alone.
