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

`micro` is `crossings layers ops contracts`. Records go to `--out`
(appended) or standard output; a readable line per case goes to standard
error.

| option | |
|---|---|
| `--rounds N` | timed rounds, 5 by default (1 for `e2e`) |
| `--warmup N` | untimed rounds first, 1 by default (none for `e2e`) |
| `--scale X` | multiplies the micro suites' repetitions |
| `--device cuda` | where the `ops` suite runs; the others run on the CPU |
| `--only 05,09` | examples for `e2e` |
| `--full` | the examples' full settings instead of the short ones |
| `--cache DIR` | one dataset cache for every example (`DIR/mnist`, ...) |
| `--variant NAME:VAR=VALUE,...` | repeat for an A/B run of `e2e` |
| `--max-load L` | wait for a 1-minute load average below `L` before each group |

CUDA runs need the CUDA shell, `env -u LD_LIBRARY_PATH nix develop .#cuda`.
A CPU run with one thread sets `OMP_NUM_THREADS=1`; `taskset -c` pins it.

## How a case is timed

Each round runs every case of a group once, rotating their order, so two
variants alternate. A case is timed with
`current-inexact-monotonic-milliseconds` over its repetitions, after its
untimed setup (`reclaim-native-memory!` for tensor cases). The record keeps
every round's per-call time and `current-gc-milliseconds` delta, their
median, quartiles and IQR, and the load average before and after.
`paired-ratio` in `harness.rkt` turns two alternating variants into a
median ratio with a sign-test interval.

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
  drain's cost per dead tensor.
- **ops** — `add` 8x8, `matmul` 64 and 512, a 3-to-16-channel `conv2d` on
  one 32x32 image, and a 64-wide `Linear` forward and backward, each through
  the facade, unchecked (the defining module's binding) and raw (the
  allocator-wrapped binding, no shape readback). On CUDA each round ends
  with an `item`.
- **contracts** — what each validation strategy costs, from AGENTS.md's
  "What the strategies cost", plus `log-mel-spectrogram` on the speech
  fixture.
- **e2e** — the examples' runners in a child process each, with their
  `env-number` overrides. Every optimizer step logs a `rktorch-step`
  event; the epochs after the first are timed from the last step of each,
  so start-up, downloads, the first epoch and the closing sampling are left
  out. Steps per second come from the median interval between steps. The
  first optimizer to step is the one timed.

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
other checkout's `torch`.

## Records

One JSON object per line: `suite` (`micro`, `e2e`), `group`, `case`,
`variant`, `unit`, `reps`, `samples` (one per round), `gc_ms`, `stats`
(`median` `q1` `q3` `iqr` `min` `max` `n`), `load` and `load_end` (1, 5
and 15-minute averages), and `meta`: `git_sha`, `git_dirty`, `racket`,
`libtorch`, `device`, `gpu`, `threads` and where that came from, `cores`,
`cpu`, `affinity`, `host`, `os`. An `e2e` run also writes one `run` record
per child: `steps`, `steps_per_epoch`, `epoch_s`, `epoch_gc_ms`, `step_ms`,
`steps_per_s`, `steps_per_s_wall`, `wall_s`, `exit_code`, `settings`.

## Baseline

`baseline-2026-10-07.jsonl` and its summary `baseline-2026-10-07.md`.
