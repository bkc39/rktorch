# Benchmarks

`scripts/bench/` measures what a native call, an op and a training epoch
cost (#252). Results are published here as JSON lines plus a generated
summary. They are never a CI gate: CI runs only `raco test scripts/bench/`,
the harness's own tests and a smoke pass of every suite.

## Running

```bash
racket scripts/bench/run.rkt micro --out results.jsonl
racket scripts/bench/run.rkt ops --device cuda --out results.jsonl
racket scripts/bench/run.rkt e2e --cache ~/.racket/rktorch --out results.jsonl
racket scripts/bench/summary.rkt results.jsonl > results.md
```

`micro` is `crossings layers ops contracts`; `pytorch-ops`, `threads`,
`pytorch-threads`, `calls` and `host-gpu` are the L0 suites (#254), and
`scripts/bench/l0-baseline.sh` runs them all (see [L0 tables](#l0-tables)).
Options go before or after
the suite names, and an unknown suite stops the run before anything is
measured. Records go to `--out` (appended) or standard output; a readable
line per case goes to standard error.

| option | |
|---|---|
| `--rounds N` | timed rounds; 5 by default for the micro suites, 1 for `e2e` |
| `--warmup N` | untimed rounds first, 1 by default and 2 for `calls`; `0` runs none, except in the scaling suites (`threads`, `pytorch-threads`), which always run one; ignored by `e2e`, which runs none |
| `--scale X` | multiplies the micro suites' repetitions |
| `--device cuda` | where `ops`, `pytorch-ops`, `calls` and `host-gpu` run (`cpu`, `cuda` or `mps`); the others run on the CPU |
| `--only 05,09` | examples for `e2e`, by prefix; `py-05` is a PyTorch twin |
| `--full` | the examples' full settings instead of the short ones |
| `--cache DIR` | one dataset cache for every example (`DIR/mnist`, ...) |
| `--variant NAME:VAR=VALUE,...` | repeat for an A/B run of `e2e`; names must differ |
| `--max-load L` | wait for a load below `L` before each group |
| `--time-limit S` | kill an `e2e` child after `S` seconds; an hour by default, twelve with `--full` |

CUDA runs need the CUDA shell, `env -u LD_LIBRARY_PATH nix develop .#cuda`.
A CPU run with one thread sets `OMP_NUM_THREADS=1`; `taskset -c` pins it.

`--max-load` samples the busy cores from `/proc/stat` over one second
(the 1-minute load average where there is no `/proc/stat`). The load
average would still count the previous example's child for a minute after
it exits.

Each `e2e` child runs in its own process group under the time limit, and
the group is killed when the child exits, so a background job it leaves
does not run into the next case; a terminal with `stty tostop` stops a
child that writes to it.

## How a case is timed

Each round runs every case of a group once, rotating their order, so two
variants alternate. A case is timed with
`current-inexact-monotonic-milliseconds` over its repetitions, after its
untimed setup: for `ops`, a collection that waits for the finalizers to
free the previous run's tensors and leaves the device's caching allocator
warm; for the other tensor cases, `reclaim-native-memory!`, which also
empties it. The record keeps
every round's per-call time and `current-gc-milliseconds` delta, their
median, quartiles and IQR, and the load average before and after.

## Suites

- **crossings** — one native call per signature class, bound with
  `get-ffi-obj` and no fault latch: `double(double,double)` from libm, a
  scalar return, a plain and a tagged pointer, a tensor struct through
  `prop:cpointer`, a `_ptr o` out-parameter, an `_s64vector`, a 4-element
  `_list`, and a `#:blocking? #t` call.
- **layers** — an 8x8 `add` with one layer of the op stack added per row:
  the `_fun` call (with an explicit free), the fault latch, the allocator
  and its finalizer, accounting and the pressure check, the OOM retry,
  shape readback, dispatch and the facade contract. `finalization` is the
  drain's cost per dead tensor; the round fails unless the timed drain
  frees every one.
- **ops** — `add` 8x8, `matmul` 64 and 512, a 3-to-16-channel `conv2d` on
  one 32x32 image, and a 64-wide `Linear` forward and backward, each through
  the facade, unchecked (the defining module's binding) and raw (the
  allocator-wrapped binding, no shape readback). On CUDA each round ends
  with an `item`.
- **contracts** — what each validation strategy costs, from AGENTS.md's
  "What the strategies cost", plus `log-mel-spectrogram` on the speech
  fixture, `--scale` calls per round. Each strategy is a submodule, so a
  `cross` loop crosses a module boundary and an `intra` loop does not. Callees come from their defining
  modules, never the contracted `torch` facade. Arguments arrive as
  parameters, since a literal lets the compiler fold an uncontracted callee
  away, and the shape loop cycles through eight tensors so the accessor is
  not hoisted out of the loop. Its `getters` group calls each short getter
  (`device`, `dtype`, `shape`, `numel`) and its `tensor-` alias on one 8x8
  tensor through the `torch` facade, so a contract that makes one name
  dearer than the other shows as a gap between the pair.
- **e2e** — the examples' runners in a child process each, with their
  `env-number` overrides. The child compiles the runner under its own
  environment before its clock starts. Every built-in optimizer's `step!`
  logs a `rktorch-step` event, and while the child listens a step on an
  accelerator waits for the device first, so the times are the device's.
  The epochs after the first are timed from the last step of each, so
  start-up, downloads, the first epoch and the closing sampling are left
  out. Steps per second are the measured steps over their span, so GC
  pauses and periodic syncs count; the median interval is kept beside it.
  The first optimizer to step is the one timed, except in 15-finetune,
  where it is the last to start (the fine-tune phase).
  `loader` is a workload, not an example: `scripts/bench/workloads/loader.rkt`
  takes one tiny step per batch, so its steps per second are the loader's
  batches per second (#168). `DATASET=folder` decodes hymenoptera's training
  JPEGs in the shim and needs `--cache`; `DATASET=racket` does
  `DECODE_WORK` passes (4, about 2 ms) of pure-Racket work per sample over
  `SAMPLES` (2048). `WORKERS` (0), `PREFETCH` (2) and `BATCH` (16) set the
  loader.
  `17-workers` is `scripts/bench/workloads/workers.rkt`: example 17's
  convnet trained from a loader with `WORKERS` (0) workers and batches of
  `BATCH` (64). `VARIANT=pgm` is the chapter's pure-Racket decode (plain-PGM
  text, rotated and shifted per sample); `VARIANT=crop` is the torch
  random-crop dataset of commit `2b7ef04` (pad 2, a random 28x28 window).
  `DATASET=fixture` trains on the committed MNIST fixture instead of the
  cache.
  `py-05-mnist`, `py-09-resnet` and `py-17-workers` are PyTorch twins in
  `scripts/bench/python/`, run by `python3` in place of `racket` and timed
  the same way: they write the steps file the `rktorch-step` listener
  writes, synchronizing the device before each step, and their GC times are
  CPython's collector pauses (`gc.callbacks`). Each twin keeps the
  example's model, batch, optimizer algorithm and hyperparameters, schedule
  and augmentation: device-resident
  data batched from a CPU `randperm` like rktorch's loader over a tensor
  dataset, `cudnn.benchmark` off, the loss read every step where the example
  reads it; 09's twin crops and flips each image as `random-crop` and
  `random-horizontal-flip` do, under `bf16` autocast on CUDA, with
  `OneCycleLR(cycle_momentum=False)` for `one-cycle-lr`. They read the
  rktorch dataset caches (`RKTORCH_MNIST_DIR`, `RKTORCH_CIFAR10_DIR`, or
  under `--cache`), and take `DATASET=fixture`, `BATCH` and (09) `BASE`.
  `py-17-workers` is `DataLoader(num_workers=WORKERS)` over the same two
  datasets, its PGM decode and jitter in pure Python.
  The twins' optimizers are PyTorch's defaults: on CUDA `Adam` and `SGD`
  pick the `foreach` kernels, one launch per op over every parameter, and
  `zero_grad()` drops the gradients (`set_to_none=True`), as `zero-grads!`
  does. rktorch's optimizers step each parameter alone, so each twin has a
  `-per-param` form (`py-05-mnist-per-param`, ...) that runs with
  `OPTIMIZER=per-param`: `foreach=False`. Both record `optimizer` and
  `foreach` (as PyTorch resolved it) in their `meta`.
  Every step is settled before it is logged. A twin calls
  `torch.cuda.synchronize()`, one call; rktorch's `settle!` reads one
  element of a parameter (`flatten`, `select`, `item`: three accounted ops
  and a device-to-host copy, since rktorch binds no stream sync), so each
  rktorch step carries a few µs of host work and a 4-byte copy the twin's
  does not.
  The recorder child reads `RKTORCH_BENCH_COLLECT_AT_TROUGHS` (`0` or `1`)
  and `RKTORCH_BENCH_COLLECT_BUDGET` (a share in (0, 1]) and runs the
  example under `native-collect-at-troughs` and `native-collect-budget` set
  to them, so a variant can sweep the GC policy. The budget is read only
  at troughs, so a budget with `RKTORCH_BENCH_COLLECT_AT_TROUGHS=0` is
  refused. A child sees these two only as its variant sets them, never from
  the caller's environment. A `torch` without
  `torch/foreign/raw/pressure-settings.rkt` runs with its own collection
  defaults, and a variant that sets either variable against it is refused.
- **pytorch-ops** — `ops`'s cases in PyTorch (`scripts/bench/python/ops.py`),
  same shapes and repetitions, eager, `variant` `pytorch`, with
  `torch.set_num_threads` set to rktorch's `at::get_num_threads`. On CUDA
  each round ends with `torch.cuda.synchronize()`. `ops` and `pytorch-ops`
  both include `matmul-2048` and a batch-128 `conv2d`. Neither side empties
  the caching allocator between runs, so both time an op on a warm cache,
  as a training loop runs it: PyTorch frees a result when its last
  reference goes, and `ops` collects and drains before every run, so each
  run starts with the previous run's blocks back in the cache. Emptying it
  instead (what `reclaim-native-memory!` does) would make every rktorch run
  pay fresh device allocations the twin's does not.
- **threads** — the ops per second of a fixed CPU op mix (an 8x8 add, a
  64-wide matmul, a relu and a sum) on 1, 2, 4 and 8 `(thread thunk #:pool
  'own)` threads, for `--scale` × 2 s a round, while the main thread is
  `idle` (blocked in `sync`) or `train`s a small MLP (784-256-10, SGD,
  batch 64), whose steps per second are kept as `main_steps_per_s`. The
  workers start, wait at a flag, and the clock starts as it releases them.
  Each worker counts every mix it finishes, and the ops are those counted
  when the main loop ends. Every worker runs its mix at least once, even
  when the run ends before it is first scheduled, so a mix that raises
  always fails the run. A round is at least 0.5 s and the scaling suites
  always run at least one warmup round, a smoke run included. The workers stop at a flag, set even when the main
  loop raises, and are killed after a grace period, never shut down with
  their custodian (racket/racket#5614); `stopped` counts those that
  stopped in time. A worker that raises stops the others, and the run
  raises it, so the case fails rather than reporting fewer ops. Warmup
  runs are not kept.
- **pytorch-threads** — the same mix and main loop in PyTorch, on
  `threading.Thread`s, with `torch.set_num_threads` at rktorch's thread
  count, released and counted the same way, and failing the same way. Its
  main loop drops the gradients (`zero_grad()`), as `zero-grads!` does.
- **calls** — native crossings per training step of 05 and 09 by phase
  (`data`, `zero-grads`, `forward`, `backward`, `optimizer`, and 09's
  `loss-item`) and in all (`step`, counted whole), on synthetic
  device-resident data (`scripts/bench/workloads/phases.rkt`), with the
  default device set outside the count, as the examples set it once per
  run. Every binding enters the fault latch, where `with-crossing-hook`
  counts the crossings. The `ledger-entry-added` race point counts the
  accounted tensors, every output of a multi-output op included, and
  `free-released` and `finalizer-releasing` count the frees. Accounting a
  tensor reads its size and device (two crossings), so a phase's op calls
  are its crossings less its frees and twice its accounted tensors; a
  tensor whose size or device cannot be read costs its two crossings
  uncounted. Counting runs with `native-collect-at-troughs` off, so no
  trough collection lands in `backward`; each counted step follows
  `reclaim-native-memory!`, so frees of an earlier step's tensors land
  outside it, and a free the finalizer thread makes lands in whichever
  phase is running.
  The crossing hook runs inside every native call, the finalizer thread's
  included, so a hook must not raise, must not call native code, and must
  tolerate being called from another thread; the bench's counters are
  `box-cas!` loops.
- **host-gpu** — 09's step on `--device`, timed per phase by the host clock
  without a sync and whole with a sync at either end: `item` reads the loss
  every step as the example does, `no-item` skips it, `phase-sync` settles
  after every phase. An unsynced phase time is not the host's own: once
  the launch queue is full the host waits on the device inside the next
  launch, so near the balance of host and device a step times about the
  same either way. A synced phase time includes `settle!`'s own cost and
  whatever the device had queued. The host's own time comes from a proxy
  instead: the same step at `#:base 4` (variant `proxy`) and the same
  batch, so the same ops and the same per-image crops, with almost no
  device work. Its step time is the host's time at the real op count; the
  `item` mode's `loss-item` wait is the device's excess over the host.
  Every step record also keeps `gc_ms`, the `current-gc-milliseconds`
  spent in its steps, per step: the proxy's smaller activations trigger
  fewer byte-driven collections than the real step, so the step less the
  proxy is device time only net of GC.
  CUDA events or `nsys` would separate the two directly; neither is bound
  or in `.#cuda`. Each warmup round runs the step and the proxy once,
  untimed; with `--warmup 0`, as in a smoke run, none runs, so the first
  round's first mode pays the first-call costs (cuDNN, the caching
  allocator's growth).

| example | short | full | timed |
|---|---|---|---|
| 05-mnist | `EPOCHS=3` | `EPOCHS=3` | epochs 2-3 |
| 06-gpt | `STEPS=300` | `STEPS=2000` | steps after 50 |
| 08-diffusion | `EPOCHS=2` | `EPOCHS=10` | epochs after the first |
| 09-resnet | `EPOCHS=3` | `EPOCHS=30` | epochs after the first |
| 10-dcgan | `EPOCHS=3` | `EPOCHS=5` | the discriminator's steps |
| 11-vae | `EPOCHS=3` | `EPOCHS=10` | epochs after the first |
| 12-char-rnn | `EPOCHS=3` | `EPOCHS=30` | epochs after the first |
| 13-translation | `EPOCHS=3` | `EPOCHS=30` | epochs after the first |
| 15-finetune | `FINETUNE_EPOCHS=3` | `FINETUNE_EPOCHS=15` | the fine-tune phase (`FEATURE_EPOCHS=0`) |
| 16-style-transfer | `STEPS=100` | `STEPS=1000` | steps after 20 |

An A/B run alternates its variants every round, and each variant is a set
of environment variables; `PLTUSERHOME=<checkout>/.racket-user` runs the
other checkout's `torch`. The recorder loads only what that `torch` has: a
checkout without `torch/foreign/raw/native-location.rkt` has its shim read from
its `native-libs/`. A checkout older than the `rktorch-step` event
(its `torch` has neither `torch/private/step-log.rkt` nor its compiled `.zo`) logs no steps, so a child of
such a checkout that completes is timed whole, from the end of its compile
to its exit (`timing` `wall`, no `error`). A child whose `torch` has the
hook, or that does not name its `torch`, and logs no steps is an `error`,
never a whole-run time. So is a child that exits non-zero, even after
logging every step, and one still running after its time limit, which is
killed with its process group. When any
run of an example is timed whole, every variant's summary for it uses the
whole-run times (`timing` `wall`, unit `s/run`), so the ratio compares like
with like; otherwise `timing` is `steps`. Relative paths in `PLTUSERHOME`, `PLTADDONDIR`
and the `RKTORCH_*_DIR` cache variables are taken from the working
directory. The summary compares each variant with the first by
`paired-ratio`: the median of the per-round time ratios (below 1 is
faster) with a sign-test interval, which needs 6 rounds to reach 95%.

## L0 tables

`scripts/bench/summary.rkt` adds a table for each L0 measurement it finds:

- **A. Per op against PyTorch** — `ops` facade and raw beside
  `pytorch-ops`, in µs per call, and their ratios, one table per device and
  thread count and invocation (`run_id`).
- **B. Thread scaling** — ops per second at each thread count, each
  implementation's ops over its own idle-main one-thread ops, rktorch over
  PyTorch, and the main thread's steps per second, one table per thread
  count and invocation (`run_id`). An idle row's ratio is
  the speedup over one thread. A train row measures how far a training
  main slows the workers: rktorch's workers run their ops on their own OS
  threads (#296), sharing only the collector with main, PyTorch's wait on
  the GIL. Before #296 rktorch train rows sat near 0 ops/s, every
  accounting waiting on main's non-yielding foreign calls (#262).
- **C. Loader end to end against PyTorch** — `17-workers` beside
  `py-17-workers` and `py-17-workers-per-param`, per variant, as D.
- **D. Training against PyTorch** — `05-mnist` and `09-resnet` beside their
  twins: the e2e median and the median step interval (`step_ms`) of each,
  rktorch over PyTorch's default optimizer and over its per-parameter one.
  A pair given a setting that one side reads and the other does not (the
  records' `reads`) gets no ratios, and the row names those settings; so
  does a twin timed in another `timing` or `unit` (a whole-run `s/run`
  against `s/epoch`), named as `PyTorch-unit=` or `per-param-unit=`. C and D
  pair only within one invocation (`run_id`).
- **E. Native calls per step** — the `calls` records: crossings, frees,
  accounted tensors and op calls.
- **F. Host and device time per step** — the `host-gpu` records: each
  phase's unsynced and synced time, their sum without `loss-item`, the
  synced step, the GC time in each step and the step net of it, the host
  proxy raw and net of GC and its share of the step either way, and the
  `loss-item` wait. Times are the host's wall clock; GC counts whole
  milliseconds a round.
- **G. GC per epoch** — every rktorch e2e summary's median `epoch_gc_ms`,
  steps per second, and GC's share of the epoch.

`scripts/bench/l0-baseline.sh` takes the baseline from the host shell,
entering one dev shell at a time: in `nix develop`, A on the CPU at the
default thread count and at `OMP_NUM_THREADS=1`, and B at one ATen thread;
in `env -u LD_LIBRARY_PATH nix develop .#cuda`, A, E and F on CUDA, C over
`VARIANT` `pgm`/`crop` × `WORKERS` 0/2/4, D at the short settings, and G
(05 and 09) with troughs off, then on at collect budgets 0.01, 0.05 (the
default), 0.2 and 1. An example and its twins run as one group, whose
order rotates every round as the micro suites' cases do, so drift reaches
both sides of a ratio alike. Each group waits for a load below 4; the
records go to `baseline-<date>.jsonl` in `BASELINE_DIR`
(`${XDG_DATA_HOME:-~/.local/share}/rktorch/bench`, outside the repository)
and the summary beside it.

```bash
scripts/bench/l0-baseline.sh         # all: cpu, cuda, summary
scripts/bench/l0-baseline.sh cpu     # one stage: cpu, cuda or summary
BASELINE_SMOKE=1 scripts/bench/l0-baseline.sh
```

`BASELINE_ROUNDS` sets every suite's rounds (unset, e2e runs 3 and the
others their own defaults); `BASELINE_MAX_LOAD` (4), `BASELINE_CACHE`
(`~/.racket/rktorch`), `BASELINE_DATE`, `BASELINE_DIR` and `BASELINE_OUT`
(the path without its extension) change the defaults. The full run starts
the JSONL afresh; a single stage appends to it. `BASELINE_SMOKE=1` writes
`smoke-<date>` and runs every step at the smallest settings both sides of
a pair read: the loader on the MNIST fixture, 05 and 09 on the cached datasets at
`EPOCHS=2` (their rktorch runners read nothing else), and G at one budget.

## Records

One JSON object per line: `suite` (`micro`, `e2e`, `scaling`, `calls`,
`host-gpu`), `impl` (`rktorch` or `pytorch`), `group`, `case`,
`variant`, `unit`, `reps`, `samples` (one per round), `gc_ms`, `stats`
(`median` `q1` `q3` `iqr` `min` `max` `n`), `load` and `load_end` (1, 5
and 15-minute averages), and `meta`: `git_sha`, `git_dirty`, `racket`,
`libtorch`, `device`, `gpu` (the selected CUDA device's name, null unless
`CUDA_VISIBLE_DEVICES` names it by a unique UUID prefix, or by index under
`CUDA_DEVICE_ORDER=PCI_BUS_ID` or on a host whose GPUs are all one model), `threads`
and where that came from (`at::get_num_threads`, else `MKL_NUM_THREADS`,
`OMP_NUM_THREADS` or the core count), `omp_num_threads` and
`mkl_num_threads` as set, `cores`, `cpu`, `affinity`, `host`, `os`; `run_id` is shared by every record one
`scripts/bench/run.rkt` invocation writes, and the summary pairs A/B ratios only within
one. The summary gives each micro invocation its own section, headed by its
commit, host, execution (device, threads and their source, GPU) and run id,
so appended runs never merge. `e2e` sections are per host (name, CPU, OS,
affinity) with the execution per row, so an A/B over `OMP_NUM_THREADS` or
the device still yields its ratio. An `e2e` run also writes one `run` record
per child: `steps`, `steps_per_epoch`, `epoch_s`, `epoch_gc_ms`, `step_ms`,
`steps_per_s`, `steps_per_s_median`, `wall_s`, `timing`, `exit_code`, `settings`
(the summary shows them without the `RKTORCH_*_DIR` caches); its `summary`
record adds each run's mean `epoch_gc_ms` and median `step_ms`, and
`reads`, the settings its runner reads. A twin's `meta` is the parent's
with the child's `impl`, `python`, `libtorch` (the torch version),
`device`, `threads` (`torch.get_num_threads`) and, for the e2e twins,
`optimizer` and `foreach` over it; torch's name for the GPU
goes to `torch_gpu`, so `gpu` stays the parent's and a twin pairs with the
rktorch records of its device. `scaling` records add `threads_run`,
`main_steps_per_s` and `stopped` (one per round) and `settings` (`seconds`
per round, `warmup` rounds); `host-gpu` step records add `gc_ms` per step,
one per round; `calls` records count
crossings in `samples`, frees in `frees` and `free_stats`, accounted
tensors in `tensors` and `tensor_stats`, and op calls in `op_calls` and
`op_call_stats`. An
`e2e` record's `meta` comes from the child that ran it, so an A/B across
checkouts records each one's commit, and the summary shows it per row.
A child sees none of the caller's example variables (`EPOCHS`, `STEPS`,
`EXCERPT`, `OUT`, ... as its runner reads them) except those its settings
give. A failed example writes an `error` record (`case`, `variant`,
`exit_code`, `error`, and `meta_from`: `child`, or `parent` when the child
died before recording its own), listed under Failures in the summary with
its commit and execution. A failed child is not run again; the rest of
its group (the example's other variants and its twins) runs every round,
and the completed runs still get their summaries.

## Baseline

Results are not committed. A leg's review packet carries its tables, and
it compares against its base by running both checkouts in one run (see
`PLTUSERHOME` above) rather than against stored records. The L0 tables are
on #380.
