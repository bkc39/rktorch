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
| `--warmup N` | untimed rounds first, 1 by default; ignored by `e2e`, which runs none |
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
untimed setup (`reclaim-native-memory!` for tensor cases). The record keeps
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
  example's model, batch, optimizer, schedule and augmentation: device-resident
  data batched from a CPU `randperm` like rktorch's loader over a tensor
  dataset, `cudnn.benchmark` off, the loss read every step where the example
  reads it; 09's twin crops and flips each image as `random-crop` and
  `random-horizontal-flip` do, under `bf16` autocast on CUDA, with
  `OneCycleLR(cycle_momentum=False)` for `one-cycle-lr`. They read the
  rktorch dataset caches (`RKTORCH_MNIST_DIR`, `RKTORCH_CIFAR10_DIR`, or
  under `--cache`), and take `DATASET=fixture`, `BATCH` and (09) `BASE`.
  `py-17-workers` is `DataLoader(num_workers=WORKERS)` over the same two
  datasets, its PGM decode and jitter in pure Python.
  The recorder child reads `RKTORCH_BENCH_COLLECT_AT_TROUGHS` (`0` or `1`)
  and `RKTORCH_BENCH_COLLECT_BUDGET` (a share in (0, 1]) and runs the
  example under `native-collect-at-troughs` and `native-collect-budget` set
  to them, so a variant can sweep the GC policy.
- **pytorch-ops** — `ops`'s cases in PyTorch (`scripts/bench/python/ops.py`),
  same shapes and repetitions, eager, `variant` `pytorch`, with
  `torch.set_num_threads` set to rktorch's `at::get_num_threads`. On CUDA
  each round ends with `torch.cuda.synchronize()`. `ops` and `pytorch-ops`
  both include `matmul-2048` and a batch-128 `conv2d`.
- **threads** — the ops per second of a fixed CPU op mix (an 8x8 add, a
  64-wide matmul, a relu and a sum) on 1, 2, 4 and 8 `(thread thunk #:pool
  'own)` threads, for `--scale` × 2 s a round, while the main thread is
  `idle` (blocked in `sync`) or `train`s a small MLP (784-256-10, SGD,
  batch 64), whose steps per second are kept as `main_steps_per_s`. The
  workers stop at a flag and are killed after a grace period, never shut
  down with their custodian (racket/racket#5614).
- **pytorch-threads** — the same mix and main loop in PyTorch, on
  `threading.Thread`s, with `torch.set_num_threads` at rktorch's thread
  count.
- **calls** — native crossings per training step of 05 and 09 by phase
  (`data`, `zero-grads`, `forward`, `backward`, `optimizer`, and 09's
  `loss-item`) and in all (`step`), on synthetic device-resident data
  (`scripts/bench/workloads/phases.rkt`). Every binding enters the fault
  latch, and a private hook there counts the crossings; the `op-returned`
  race point counts the accounted tensors. Each counted step follows
  `reclaim-native-memory!`, so frees of an earlier step's tensors land
  outside it; frees of its own temporaries land in a later step.
- **host-gpu** — 09's step on `--device`, timed per phase by the host clock
  without a sync (the host's enqueue time) and whole with a sync at either
  end: `item` reads the loss every step as the example does, `no-item`
  skips it, `phase-sync` syncs after every phase so each phase's time is
  the device's. A step whose host enqueue time is close to its synced time
  is bound by the host.

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
other checkout's `torch`. A checkout older than the `rktorch-step` event
(its `torch` has neither `private/step-log.rkt` nor its compiled `.zo`) logs no steps, so a child of
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

`summary.rkt` adds a table for each L0 measurement it finds:

- **A. Per op against PyTorch** — `ops` facade and raw beside
  `pytorch-ops`, in µs per call, and their ratios, one table per device and
  thread count.
- **B. Thread scaling** — ops per second at each thread count, the speedup
  over one thread, rktorch over PyTorch, and the main thread's steps per
  second.
- **C. Loader end to end against PyTorch** — `17-workers` beside
  `py-17-workers`, per variant.
- **D. Training against PyTorch** — `05-mnist` and `09-resnet` beside their
  twins: the e2e median, the median step interval (`step_ms`), and their
  ratios.
- **E. Native calls per step** — the `calls` records.
- **F. Host and device time per step** — the `host-gpu` records: each
  phase's host time, their sum without `loss-item` (the host's enqueue
  time), the synced step, and the share of the step the host spends
  enqueueing. Times are the host's wall clock (`nsys` is not in the CUDA
  shell).
- **G. GC per epoch** — every rktorch e2e summary's median `epoch_gc_ms`,
  steps per second, and GC's share of the epoch.

`scripts/bench/l0-baseline.sh` takes the baseline from the host shell,
entering one dev shell at a time: in `nix develop`, A on the CPU at the
default thread count and at `OMP_NUM_THREADS=1`, and B at one ATen thread;
in `env -u LD_LIBRARY_PATH nix develop .#cuda`, A, E and F on CUDA, C over
`VARIANT` `pgm`/`crop` × `WORKERS` 0/2/4, D at the short settings, and G
(05 and 09) with troughs on, then off at collect budgets 0.01, 0.05, 0.2
and 1. Each group waits for a load below 4; the records go to
`docs/benchmarks/baseline-<date>.jsonl` and the summary beside it. `cpu`,
`cuda` or `summary` runs one stage; `BASELINE_ROUNDS` (3),
`BASELINE_MAX_LOAD`, `BASELINE_CACHE` (`~/.racket/rktorch`),
`BASELINE_DATE` and `BASELINE_OUT` change the defaults, and
`BASELINE_SMOKE=1` with `BASELINE_OUT` runs every step at the smallest
settings, on the fixtures where a runner takes them.

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
`run.rkt` invocation writes, and the summary pairs A/B ratios only within
one. The summary gives each micro invocation its own section, headed by its
commit, host, execution (device, threads and their source, GPU) and run id,
so appended runs never merge. `e2e` sections are per host (name, CPU, OS,
affinity) with the execution per row, so an A/B over `OMP_NUM_THREADS` or
the device still yields its ratio. An `e2e` run also writes one `run` record
per child: `steps`, `steps_per_epoch`, `epoch_s`, `epoch_gc_ms`, `step_ms`,
`steps_per_s`, `steps_per_s_median`, `wall_s`, `timing`, `exit_code`, `settings`
(the summary shows them without the `RKTORCH_*_DIR` caches); its `summary`
record adds each run's mean `epoch_gc_ms` and median `step_ms`. A twin's
`meta` is the parent's with the child's `impl`, `python`, `libtorch` (the
torch version), `device`, `gpu` and `threads` (`torch.get_num_threads`)
over it. `scaling` records add `threads_run` and `main_steps_per_s` (one
per round); `calls` records count crossings in `samples` and accounted
tensors in `tensors` and `tensor_stats`. An
`e2e` record's `meta` comes from the child that ran it, so an A/B across
checkouts records each one's commit, and the summary shows it per row.
A child sees none of the caller's example variables (`EPOCHS`, `STEPS`,
`EXCERPT`, `OUT`, ... as its runner reads them) except those its settings
give. A failed example writes an `error` record (`case`, `variant`,
`exit_code`, `error`, and `meta_from`: `child`, or `parent` when the child
died before recording its own), listed under Failures in the summary with
its commit and execution. A failure ends that example's rounds; the
variants' completed runs still get their summaries.

## Baseline

`baseline-2026-10-13.jsonl` and its summary `baseline-2026-10-13.md`, taken at
`f8b9014` on the lab host (i9-9900K, RTX 3090 Ti, 8 ATen threads): `micro`,
`ops --device cuda` and the short `e2e`, each started once the 1-minute load
was below 4.
