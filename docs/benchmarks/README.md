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

`micro` is `crossings layers ops contracts`. Options go before or after
the suite names, and an unknown suite stops the run before anything is
measured. Records go to `--out` (appended) or standard output; a readable
line per case goes to standard error.

| option | |
|---|---|
| `--rounds N` | timed rounds; 5 by default for the micro suites, 1 for `e2e` |
| `--warmup N` | untimed rounds first, 1 by default; ignored by `e2e`, which runs none |
| `--scale X` | multiplies the micro suites' repetitions |
| `--device cuda` | where the `ops` suite runs (`cpu`, `cuda` or `mps`); the others run on the CPU |
| `--only 05,09` | examples for `e2e` |
| `--full` | the examples' full settings instead of the short ones |
| `--cache DIR` | one dataset cache for every example (`DIR/mnist`, ...) |
| `--variant NAME:VAR=VALUE,...` | repeat for an A/B run of `e2e`; names must differ |
| `--max-load L` | wait for a load below `L` before each group |

CUDA runs need the CUDA shell, `env -u LD_LIBRARY_PATH nix develop .#cuda`.
A CPU run with one thread sets `OMP_NUM_THREADS=1`; `taskset -c` pins it.

`--max-load` samples the busy cores from `/proc/stat` over one second
(the 1-minute load average where there is no `/proc/stat`). The load
average would still count the previous example's child for a minute after
it exits.

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
  not hoisted out of the loop.
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
(its `torch` has no `private/step-log.rkt`) logs no steps, so its child is
timed whole, from the end of its compile to its exit (`timing` `wall`, no
`error`). A child whose `torch` has the hook and logs no steps is an
`error` naming that `torch`, never a whole-run time. When any
run of an example is timed whole, every variant's summary for it uses the
whole-run times (`timing` `wall`, unit `s/run`), so the ratio compares like
with like; otherwise `timing` is `steps`. Relative paths in `PLTUSERHOME`, `PLTADDONDIR`
and the `RKTORCH_*_DIR` cache variables are taken from the working
directory. The summary compares each variant with the first by
`paired-ratio`: the median of the per-round time ratios (below 1 is
faster) with a sign-test interval, which needs 6 rounds to reach 95%.

## Records

One JSON object per line: `suite` (`micro`, `e2e`), `group`, `case`,
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
(the summary shows them without the `RKTORCH_*_DIR` caches). An
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
