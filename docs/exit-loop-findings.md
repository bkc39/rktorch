# #72 follow-up: repro attempt + root-cause findings (aarch64-darwin)

Host: Apple Silicon (T6020), macOS 25.5.0, Racket **v9.3 CS** inside `nix develop`
(the system Racket 8.17 cannot `require torch` at all). Tree: `f534742`
(post-MPS #71/#74, post-indexing #70), dylib rebuilt and re-staged from
`nix build .#cpp` before every measurement.

## Headline

**#72's stated root cause does not hold on darwin, and I could not reproduce the
cascade on current master in ~400 sessions.** Several load-bearing assumptions in
the issue turn out to be false here. Details and evidence below.

## What was ruled out (with evidence)

### 1. Exit does not run finalizers — so there is no "exit-time finalizer sweep"
```
racket -e '(require ffi/unsafe) (register-finalizer (box 0) (lambda (_) (eprintf "FIN\n"))) (void)'
```
prints nothing, on both Racket 8.17 and 9.3. Registering 3000 finalizers and
exiting drains **zero** of them. `PLTSTDERR=debug@GC` shows `0:atexit` with no
final collection. This confirms `plans/gpu-memory-management.md:21-22`
("exit skips pending finalizers") still holds, and it removes the mechanism #72
assumed: *"Exit is exactly when the process touches lots of native code (GC
finalizers over accumulated tensors)."* It isn't.

`ffi/unsafe/alloc`'s `release-all` (which *would* free every outstanding
allocation at teardown) is registered via `unsafe-add-post-custodian-shutdown`
and is explicitly **a no-op in the main place** — and a REPL is the main place.

### 2. In-place `.so`/`.dylib` overwrite is inert on darwin
macOS does **not** refuse the write (`cp`, `dd conv=notrunc`, and a plain Python
`r+b` write all succeed against a live mapping — `vmmap` confirms 4 mapped
regions). But the shim is only ~195 KB and is fully resident after load, so the
writes never reach the mapped pages. **Zeroing the entire dylib past the Mach
header under a live REPL, then calling six cold ops (`matmul`, `sum`, `reshape`,
`exp`, `eye`, `cat`), changed nothing** — every op returned correct results and
the session exited 0. No `CODE SIGNING` / `cs_invalid_page` entries in `log show`
either, despite the dylib being signed and valid.

So the fix proposed in #72 (atomic rename staging) is still *good hygiene* and
worth doing — but it cannot be what caused a darwin incident.

### 3. The finalizer guard is airtight, and universal
Audited every `define-torch` form in the tree: **33 return `_Tensor/null`, and
all 33 carry a `#:wrap`** (27 `tensor-allocator`, 3 `tensor-allocator/rng`, 3 via
the `define-*/raw` macros, plus both generated-op arms). Every `(allocator ...)`
in the tree is inside `tensor-allocator`/`tensor-allocator/rng`, i.e. behind
`swallow-and-count-failure`. There is no bare-`(allocator ...)` path.

Verified behaviourally: 20,000 deliberately faulting finalizers under a guard
identical to `swallow-and-count-failure` → all 20,000 swallowed, session clean.

### 4. Use-after-free is contained
Forging a live-looking `'Tensor` tag on a freed handle and printing it hits the
printer's `exn:fail?` guard and falls back to `#<tensor:64x64>`; an *op* on it
raises `invalid memory reference` exactly **once** and the REPL continues.
`invalid memory reference` is a plain catchable `exn:fail` (not
`exn:fail:contract`), so `structs.rkt`'s printer guard does cover it.

### 5. Negative repro matrix (all on the real runtime)
| shape | result |
|---|---|
| baseline / churn-cpu / live-cpu / print-cpu | 0/20 each |
| churn-mps / live-mps / print-mps / mixed-mps | 0/20 each |
| heavy MPS train (400 steps) + 3000 matmuls + 1200x1200 prints | clean, `finalizer-failures`=0 |
| 150 forced MPS OOMs through `oom-retry` → `collect-and-drain!` → `tr_mps_empty_cache`, tensors held live | clean, `finalizer-failures`=0 |

And on a **real pty with a real Ctrl-D** (0x04), 8 iterations per mode:

| pty mode | what it does | result |
|---|---|---|
| `idle` | Ctrl-D at a quiet prompt, 400 tensors live | 0/8 |
| `printing` | Ctrl-D while a 1200x1200 tensor is streaming | 0/8 |
| `busy` | Ctrl-D while 4000 matmuls are still running | 0/8 |
| `intr` | Ctrl-C then Ctrl-D mid-computation | 0/8 |
| `spam` | ten Ctrl-Ds in a row | 0/8 |

`finalizer-failures` was **0 in every single run**.

## What IS confirmed: the amplifier

The runaway is real and I reproduced its shape — but only by *removing* the guard.
An unguarded faulting finalizer produces **1.1 MB of repeated
`invalid memory reference.  Some debugging context lost`** from a single
`collect-garbage`, one line per dead handle.

The nested form in the report — `...; original exception raised: exception raised
by error display handler: ...; original exception raised: ...` — is Racket's
nested-exception message *concatenating the previous whole message each round*.
That is the unbounded allocation: the message grows every iteration, so printing
it is O(n²). Reproduced synthetically with a re-raising
`error-display-handler` + `error-escape-handler`. It needs a **persistent**
fault in the display/escape path, not a one-shot one.

Also worth noting for triage: `racket -i` auto-loads **xrepl** (no `~/.racketrc`
needed), so xrepl's line editor and error display are in the loop. On a tty,
Ctrl-D on a *non-empty* line is delete-char, not EOF.

## One thing that *did* reproduce "Killed" (a different bug)

`(zeros 1000000000000)` on **CPU** — 4 TB, i.e. large but *within* the 64-bit
address space — is accepted by macOS as a reservation and the process is
**SIGKILLed (exit 137) while zero-filling it**, with no crash report. The OOM
classifier never runs, so no `exn:fail:rktorch:oom` is ever raised. This is the
same "Killed, no `.ips`" signature as the reported incident, but it is host OOM,
not the error cascade. It is why the harness's OOM shape uses 2^60 floats
(beyond address space, refused outright by every backend) — the value
`oom-error-test.rkt` already uses. Worth its own issue if you want typed OOM to
cover "the host said yes and then killed us".

## Remaining leads

1. **The `mps-device` worktree** (`.claude/worktrees/mps-device`, HEAD `7534c2c`,
   dylib staged 08-23 12:00) is the likely home of the MPS REPL sessions, and
   Aug 23 is when MPS was being rebuilt repeatedly. Its dylib is internally
   consistent with its HEAD and its bytecode cache is empty, so no ABI mismatch
   is visible *now* — but a mid-session rebuild there is the one thing I cannot
   reconstruct after the fact.
2. **Date-bounding.** The finalizer guard landed `3eac103` (2026-08-17). Incidents
   before that date are explained by the pre-guard cascade (#38) and are not
   evidence about current master.
3. No `racket*.ips` crash reports exist (only expected `torchrkt_tests`
   death-test ones), which is consistent with SIGKILL and rules out a native
   SIGSEGV at exit.

## Harness added

- `scripts/repro/exit-loop.sh <shape.rkt> [n]` — pipes a shape into `racket -i`
  (EOF ≡ Ctrl-D) n times, flags nonzero exits and cascade text, reports a rate.
- `scripts/repro/shapes/*.rkt` — 10 shapes (cpu/mps × churn/live/print, heavy
  train, forced-OOM churn).
- `scripts/repro/pty-ctrl-d.py <mode> [n]` — drives `racket -i` on a **real pty**
  and sends a real Ctrl-D (0x04); modes `idle|printing|busy|intr|spam`.
- `scripts/repro/corrupt-restage.sh [zero|restage]` — the #72 vector: overwrite
  the mapped dylib under a live REPL, then EOF; restores afterwards.

---

# Addendum: x86_64-linux / CUDA host — **the cascade reproduces, first try**

Host: `lab`, i9-9900K / RTX 3090 Ti, Ubuntu (Linux 6.8.0-124), Racket **9.3 CS** inside
`nix develop`. Worktree `.claude/worktrees/exit-loop` at `8ae14fb`, CPU shim
`vgbqkn6x…-torchrkt-cpp` (221,936 B) and CUDA shim `dbpbi5x8…` (226,184 B) freshly built
and staged, bytecode cleared, `raco setup` clean.

**Headline: every negative result above is darwin-specific. On Linux the #72 vector is
live, it fires on the first attempt, and it does not need CUDA — plain CPU is enough.**

## 1. Assumption 2 is false here: an in-place overwrite DOES reach the mapping

`repro-logs/e0-mmap-propagation.py` — no torch involved. `mmap` a 256 KB file
`PROT_READ`/`MAP_PRIVATE`, touch **every** page so the mapping is fully resident (darwin's
stated reason the overwrite was inert), then modify the file three ways:

```
pre : byte[0]=b'AAAA' byte[128k]=b'AAAA' (all 64 pages faulted in)
post-write(2)      : byte[0]=b'BBBB' byte[128k]=b'BBBB'
post-cp (same size): byte[0]=b'CCCC' byte[128k]=b'CCCC'
post-cp (shorter)  : byte[0]=b'DDDD'  -> Bus error (core dumped), exit 135
```

Linux's page cache *is* the backing store for clean file-backed pages, so residency is no
protection: a `write(2)` and a `cp` are both visible through the mapping, and `cp` of a
**shorter** file leaves mapped pages past EOF and SIGBUSes the process outright.

The live REPL's own `smaps` confirms the darwin premise holds here and still doesn't help
— `Size == Rss` on every segment of the shim, all `Private_Clean`:

```
Size:  36 kB   Rss:  36 kB   Private_Clean:  36 kB   VmFlags: rd mr mw me sd
Size: 112 kB   Rss: 112 kB   Private_Clean: 112 kB   VmFlags: rd ex mr mw me sd   <- .text
Size:  32 kB   Rss:  32 kB   Private_Clean:  32 kB   VmFlags: rd mr mw me sd
```

Fully resident, and corrupted anyway.

## 2. The cascade, reproduced — `restage` mode, CPU, first run

`scripts/repro/corrupt-restage.sh restage` — 500 live 256x256 tensors, then re-copy
**identical bytes** in place (exactly `nix run .#copy-native-libs`), then seven cold ops,
then EOF:

```
[repro] overwriting mapped image in place (mode=restage)
  re-copied identical bytes
[repro] STILL ALIVE after 40s -- snapshotting
VmRSS:	  402176 kB      Threads: 1
[repro] SIGTERM
[repro] TERM IGNORED -> SIGKILL
[repro] exit=137  invalid-memory-reference hits: 20  cascade hits: 12
```

All four reported symptoms, in one run:

| #72 symptom | observed |
|---|---|
| repeated `invalid memory reference. Some debugging context lost` | 20 |
| nested `...; original exception raised: exception raised by error escape handler: ...` | 12, nesting depth 11 |
| SIGTERM ignored | yes |
| SIGKILL required | yes (exit 137) |

The transcript pins the mechanism, and it is **not** an exit-time finalizer sweep
(the darwin finding that finalizers never run at exit is correct and still stands):

```
> ; invalid memory reference.  Some debugging context lost [,bt for context]   x7
exception raised by error display handler: thread: the custodian has been shut down
  custodian: #<custodian>; original exception raised: invalid memory reference. ...
  context...:
   .../xrepl-lib/xrepl/xrepl.rkt:193:0: do-wrapped-output
invalid memory reference.  Some debugging context lost
exception raised by error escape handler: invalid memory reference. ... ; original
  exception raised: exception raised by error escape handler: ...   [nests 11 deep]
```

EOF tears down the custodian; the error **display** handler then fails because its
custodian is gone; the error **escape** handler re-enters the corrupted shim and raises
again; each round re-concatenates the whole previous message. That is precisely the
O(n^2) nesting the darwin write-up reproduced synthetically — here it arises for real,
from the staging vector.

## 3. Matrix — all Linux/CPU, all with 500 live tensors

| experiment | what was written over the mapped `.so` | IMR | cascade | hung | signal |
|---|---|---|---|---|---|
| `corrupt-restage.sh restage` | identical bytes (`cp`) | 20 | 12 | yes | TERM ignored → KILL |
| `corrupt-restage.sh zero` | whole image zeroed past 4 KB | 66 | 58 | yes | TERM ignored → KILL |
| E2 `cuda-over-cpu` | the **CUDA** shim (`cp -f`, +4,248 B) | 68 | 58 | yes | TERM ignored → KILL |
| E1c real pty + real Ctrl-D (0x04) | identical bytes (`cp`) | 84 | 66 | yes | TERM ignored → KILL |

`restage` — *identical bytes* — is the important row: nothing about the new content
matters. `cp` opens with `O_TRUNC`, and the truncation alone invalidates the mapped
pages. Every everyday staging path in the tree is that `cp`.

### On CPU the runaway wedges early; on CUDA it explodes (section 7)

`repro-logs/e1b-*-curve.txt` and `e1c-*-curve.txt` follow RSS and output after Ctrl-D. On
**CPU** the cascade runs ~11 rounds, stops, and the process sits wedged and TERM-immune
forever — RSS flat at ~430 MB, output flat at 20,424 B. The hang and the TERM-immunity
reproduce; the unbounded growth does not. It does on CUDA — see section 7.

## 4. Control: untouched master is clean

`./scripts/repro/run-all.sh cpu 20` — all twelve arms, no restage anywhere:

```
RESULT s0-baseline-cpu:   0/20      RESULT pty/idle-cpu:     0/8
RESULT s1-churn-cpu:      0/20      RESULT pty/printing-cpu: 0/8
RESULT s2-live-cpu:       0/20      RESULT pty/busy-cpu:     0/8
RESULT s3-print-cpu:      0/20      RESULT pty/intr-cpu:     0/8
RESULT s4-mixed-cpu:      0/20      RESULT pty/spam-cpu:     0/8
RESULT s5-heavy-train-cpu:0/20
RESULT s6-oom-churn-cpu:  0/20
[rktorch mem] ((runs . 2998) (failures . 0) (messages) (ledger-entries . 503))
```

2,998 native frees actually performed, zero swallowed failures. The fault appears *only*
when something rewrites the shim underneath a live process — which is what makes the
staging paths, not the runtime, the thing to fix.

## 5. The staging surface is wider than #72 assumed

One loader (`torch/foreign/raw/syntax.rkt:17-20`), package installed in **link** mode, so
that path is the worktree file. Everything that writes it:

| site | mechanism | inode | guard |
|---|---|---|---|
| `copy-native-libs` — `flake.nix:340` | `cp -v --no-preserve=mode` | **truncates in place** | none |
| `cudaHook` — `flake.nix:536-537` | `cp -f --no-preserve=mode` | **truncates in place** | **none — every `.#cuda` entry** |
| `provisionRacket` — `flake.nix:512` | `cp` + `2>/dev/null \|\| true` | silently fails on a `0444` file | `deps_stamp`, first provision only |
| `install-torchrkt-native.rkt:24-26` | `delete-file` + `copy-file` | new inode — safe, non-atomic gap | `TORCHRKT_NATIVE_LIB_PATH` → **every `raco setup --pkgs torch`** |
| `docs/exit-loop-runbook.txt:108` | `cp`, no `rm` | **truncates in place** | manual |

Two of these fire without anyone asking for a restage: **entering `.#cuda`** (unguarded,
and it swaps in a *differently sized* shim — 226,184 B vs 221,936 B, the shrinking case
that SIGBUSes), and **`raco setup --pkgs torch`**, since `TORCHRKT_NATIVE_LIB_PATH` is
exported in every nix shell. The routine post-C++-change command is itself the vector.

`cpp/CMakeLists.txt` never copies into the source tree; `.github/workflows/*` has no
staging step.

## 6. The amplifier: why one transient fault wedges the process forever

`swallow-and-count-failure` is not merely a nicety — it is the only thing in front of an
**unprotected** atomic region in Racket itself, `ffi/unsafe/alloc.rkt:29-42`:

```racket
(define (deallocate v)
  (unsafe-start-atomic)
  ... ((node-proc ds) v) ...        ; <- tr-tensor-free/finalizer
  (unsafe-end-atomic))
```

Raw start/end, no `dynamic-wind`, no handler. Anything escaping `(node-proc ds)` skips
`unsafe-end-atomic` and leaves the process atomic permanently — scheduler dead, no thread
swaps, so SIGTERM's Racket-level handling never runs and only SIGKILL ends it. That is
exactly the observed symptom.

(By contrast `call-with-ledger`'s `call-as-atomic` is exception-safe —
`ffi/unsafe/atomic.rkt:100-112` aborts out of the atomic region before handlers run — so
the ledger is not the leak.)

Two gaps in the guard, `torch/foreign/raw/memory.rkt:69-75`:

- the run-counter `call-with-ledger` at `:72-73` runs **before and outside** the
  `with-handlers`;
- `record-failure!` (`:60-67`) runs *inside* the handler with no guard of its own and
  calls `(format "~e" e)` on a non-`exn?` value — for a `tensor-impl` that re-enters
  `prop:custom-write` → `handle->repr` → FFI → possibly another fault, inside atomic mode.

And two candidates for the *persistent* display-path fault the cascade requires:
`torch/foreign/error.rkt:12-15` (`last-failure` makes native calls inside `call-as-atomic`,
so a corrupted shim faults on the error-**reporting** path), and
`torch/foreign/structs.rkt:158-172`, whose printer guard is `exn:fail?` rather than total
and whose fallback still `fprintf`s to the port — which fails when the port's custodian is
down, literally what the transcript reports.

This restores an intent already written down in `plans/gpu-memory-management.md:56`: guard
"right at the finalizer boundary, **so error handlers never loop**".

## 7. CUDA is where the unbounded growth lives

The CUDA arm is not a formality — it reproduces the half of the incident CPU could not.
Same vector, same script, only `REPRO_DEVICE=cuda`:

| arm | IMR | cascade | final nesting depth | stderr | RSS at kill | threads |
|---|---|---|---|---|---|---|
| CPU, `restage` | 20 | 12 | 11 | 9,827 B | 402 MB | 1 |
| **CUDA, `restage`** | **3,077** | **3,069** | **3,068** | **588,650,462 B** | **3.23 GB** | **4** |
| CUDA, `zero` | 3,223 | 3,215 | — | 645,990,867 B | 3.58 GB | 4 |
| CUDA, CPU-shim-over-CUDA-shim (shrinking) | 3,225 | 3,215 | 3,214 | 645,991,159 B | 3.63 GB | 4 |

And the growth curve (`repro-logs/e1b-restage-cuda-curve.txt`), which is #72's "unbounded
allocation" measured:

```
t=  0s  rss=   603,944 kB   log=       150,612 B
t=  5s  rss= 2,011,356 kB   log=   297,417,290 B
t= 10s  rss= 3,228,232 kB   log=   588,650,462 B
t= 15s+ flat -- wedged, SIGTERM ignored, SIGKILL required
```

**0.6 GB → 3.2 GB and 588 MB of stderr in ten seconds**, from re-copying identical bytes.
Each round re-concatenates the entire previous message, so printing is O(n²) — at depth
3,068 a single message is hundreds of MB. The reported 8.6 → 10.2 GB is the same curve on
a REPL with a larger starting heap.

Why CUDA and not CPU: the CUDA-linked process has **4 threads** to the CPU build's **1**.
With one thread the scheduler wedges almost immediately (11 rounds); with the CUDA
runtime's threads still runnable the error loop keeps being driven for thousands of rounds
before it finally wedges too. Both end the same way — TERM-immune, SIGKILL.

The sweep itself is clean on CUDA, exactly as on CPU: all 12 arms 0 failures,
`[rktorch mem] ((runs . 2998) (failures . 0) (messages) (ledger-entries . 503))`. So this
is not a CUDA defect — it is the same staging vector with a louder amplifier.

## 8. Fix and verification

Every staging path now writes a temp file in the destination directory and `rename(2)`s it
into place (`flake.nix` `stageNativeLibs`/`stageNativeLibsStamped`,
`torch/private/install-torchrkt-native.rkt` via `with-temporary-file` from
`torch/private/util.rkt`). `rename` swaps the directory entry and leaves the old inode
alive, so a live process keeps executing the old lib — stale-but-safe, which is the trade
#72 asked for — and there is no window where the file is absent or half-written.

Shell-entry staging is additionally keyed to a `.racket-user/.staged-shim` stamp, so
`.#cuda` no longer restages on every entry, and returning to the default shell now restores
the CPU shim instead of silently leaving the CUDA-linked one staged. `cudaHook` also
re-points `TORCHRKT_NATIVE_LIB_PATH` at `cpp-cuda`; previously it kept pointing at the CPU
`cpp`, so any `raco setup --pkgs torch` inside the CUDA shell staged the CPU shim back over
the CUDA one.

Verified by running the **real** staging commands underneath a live REPL holding 500
tensors, then exercising cold native code and sending Ctrl-D
(`repro-logs/e7-*`):

| vector | inode | staged content | result |
|---|---|---|---|
| `nix run .#copy-native-libs` | changed | same | exit 0, imr=0, cascade=0 |
| `nix develop .#cuda` entry | changed | **CPU shim → CUDA shim** | exit 0, imr=0, cascade=0 |
| `raco setup --no-docs --pkgs torch` | changed | same | exit 0, imr=0, cascade=0 |

The middle row is the decisive one: a *different, larger* binary replaced the file under a
running REPL, which kept executing the old inode and exited cleanly. Before the fix that
same scenario produced 3,225 faults and a 3.6 GB runaway.

## 9. Still open: the amplifier (separate issue)

The staging fix removes the trigger, not the amplifier. A native fault from any other cause
would still cascade, because `ffi/unsafe/alloc.rkt:29-42` runs finalizers under raw
`unsafe-start-atomic`/`unsafe-end-atomic` with no `dynamic-wind` — anything escaping leaves
the process atomic forever, which is why SIGTERM is ignored. Filed separately with four
located leads: the two gaps in `swallow-and-count-failure`
(`torch/foreign/raw/memory.rkt:69-75`), native calls inside `call-as-atomic` on the
error-reporting path (`torch/foreign/error.rkt:12-15`), the `exn:fail?`-only printer guard
whose fallback still writes to a possibly-dead port (`torch/foreign/structs.rkt:158-172`),
and a poisoned-shim latch at the single `define-ffi-definer`
(`torch/foreign/raw/syntax.rkt:19-20`).
