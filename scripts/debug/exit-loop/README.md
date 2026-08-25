# exit-loop repro harness

These scripts were written to debug **#72** — an interactive REPL that, on exit,
printed an unbounded nested error cascade, ignored SIGTERM, and had to be
SIGKILLed. Root cause: staging `libtorchrkt` with an in-place `cp` truncates the
file under any process that has it mapped. The follow-up, why one fault cascades
instead of aborting once, is **#76**.

Keep them so the failure can be re-run if it resurfaces.

## Running

```bash
nix run .#copy-native-libs                                     # stage a matching shim first
nix develop        --command ./scripts/debug/exit-loop/run-all.sh cpu  20
nix develop .#cuda --command ./scripts/debug/exit-loop/run-all.sh cuda 20
```

Reports land in `repro-logs/report-<device>.txt`; only failing transcripts are kept.

## Layout

| path | what it does |
|---|---|
| `run-all.sh <device> [n]` | whole sweep for one device into one report |
| `exit-loop.sh <shape> [n]` | one shape, n times, via piped stdin (EOF ≡ Ctrl-D) |
| `pty-ctrl-d.py <mode> [n]` | same, on a **real pty** with a real Ctrl-D (0x04) |
| `corrupt-restage.sh [zero\|restage]` | the #72 vector: overwrite the mapped shim under a live REPL |
| `preflight.rkt` | racket build, device availability, staged shim and its linkage |
| `shape-prelude.rkt` | shared device selection, prepended to every shape |
| `shapes/` | workloads: churn, live tensors, printing, training, forced OOM |

`REPRO_DEVICE` (cpu\|cuda\|mps), `REPRO_TIMEOUT` (sec), `REPRO_LOGDIR` are the knobs.

## Reading the results

A run fails on fault text (`invalid memory reference`, the error display/escape
handler cascade), on xrepl's `[,bt for context]` banner — which means the shape
itself errored and tested nothing — or on a hang that also ignores SIGTERM.

`pty/busy` may report `INCONCLUSIVE`: it sends Ctrl-D mid-computation, and on a
tty the line editor treats Ctrl-D on a non-empty line as delete-char, so the EOT
is sometimes swallowed. That is not the cascade and is counted separately.

`corrupt-restage.sh` is destructive by design. It needs the shim writable (staging
leaves it `0555`), prints the inode either side to prove the write was in place,
and restores from an EXIT trap.
