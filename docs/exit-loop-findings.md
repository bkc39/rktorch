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
