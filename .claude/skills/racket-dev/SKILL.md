---
name: racket-dev
description: Local Racket development loop for the torch bindings — compile, test, cover, lint, and build the docs for every Racket change before calling it done. Use this whenever editing, adding, or reviewing anything under torch/, examples/ or scripts/ (.rkt/.scrbl files), whenever adding a public binding, contract or layer, and whenever CI fails on the Resyntax lane or the racket check and the fix needs reproducing locally.
---

# Racket development loop (torch)

The counterpart of `cpp-dev` for the Racket side. Every gate CI runs is
runnable locally and much cheaper here than at the end of a review round: a
push costs a full CI cycle and a fresh round from both review bots.

**A Racket change is not done while any gate below is red.** The coverage
floor in particular is a gate, not a report: do not end a turn having added
library code that the suite never executes.

## The loop

### 0. Stage before any nix command

The flake builds from the **git-tracked tree**, so untracked files are
invisible to `nix develop` / `nix build` / `nix flake check`:

```bash
git add -A
```

### 1. After a C++ change, re-stage the shim first

Racket tests otherwise run against the stale library and fail with
`dlsym ... symbol not found`:

```bash
nix run .#copy-native-libs
```

### 2. Compile before testing

`raco test` does **not** recompile a module's dependents. After editing a
module that others require, stale bytecode surfaces as
`instantiate-linklet: mismatch; reference to a variable that is not
exported` — which looks like a code error and is not:

```bash
nix develop .#ci --command raco make torch/tests/*.rkt
```

If it persists, clear the caches and redo:
`find torch examples -type d -name compiled -exec rm -rf {} +`

### 3. Test

```bash
nix develop .#ci --command raco test torch/
```

Shell entry installs the package in link mode, so the `torch` collection
resolves from anywhere in the checkout without setting `PLTCOLLECTS`.
The PyTorch parity tests self-skip in `.#ci` (no wheel) and
run for real in the default `nix develop`; run them there before pushing
anything that touches an op, an initializer or an optimizer.

Examples have their own runners: `raco test examples/test/`.

### 4. Coverage — the floor is a gate

```bash
nix develop .#ci --command racket scripts/coverage.rkt --changed
```

It instruments the library, drives it with the suite, prints the total plus
the per-area table, and **exits non-zero below the floor** (`coverage-floor`
in the script; see #173 for the climb to 99%). `--changed` additionally lists
the files this branch touches with the **line numbers** that no test reaches,
which is the actionable part:

```
  torch/nn/group-norm.rkt   81.5   25 missed of 135
      uncovered lines: 25 26 27 28 29
```

Cold it takes about a minute, less than the test run it replaces, so run it
before pushing rather than after a bot asks. Coverage says a line ran, not
that anything checked it, so write the test for the behaviour and let the
number follow.

`--changed` compares against `origin/master`. In a shallow checkout there is
no merge base to compare with, so it says so and falls back to the working
tree and the index rather than failing the run.

`cover` runs every file in **one process**, where `raco test` forks per file,
so a test that asserts on accumulated ledger or GC state can fail under
coverage and pass under `raco test` — the pressure tests do. Step 3 is the
authority on whether the suite passes; the script says so and treats its own
numbers as a floor when that happens.

Legitimately unreachable here: accelerator-only branches on the wrong host
(MPS cannot be covered on Linux at all) and network download paths. Anything
else uncovered is either a missing test or, when a contract already rejects
the input, a dead guard to delete — see `torch/foreign/ops.rkt`'s
`->i` + `#:pre` pattern and #96.

### 5. Resyntax — exactly what CI runs

```bash
nix develop .#ci --command \
  resyntax analyze --local-git-repository . origin/master --analyzer-timeout 30000
```

CI fails on **any** suggestion. `resyntax fix --local-git-repository . origin/master`
applies them. Resyntax skips a file whose `compiled/` was built by a different
Racket than the shell's, and it skips `#lang scribble/lp2` files entirely, so
a clean run on the literate examples proves nothing.

### 6. raco review

```bash
nix develop .#ci --command raco review torch/foreign/ops.rkt ...
```

Known false positives: a template `(define pattern-var ...)` inside a
`begin-for-syntax` helper reads as "already defined", and a curried
`(define ((f self) stx))` header reads as binding `self`. Mark the first
`;; noqa`; name the parameter `self-id` for the second. Pure re-export
facades (`main.rkt`, `foreign.rkt`, `nn.rkt`) and `info.rkt` carry
`#|review: ignore|#`.

### 7. Docs build

Any new public binding needs a Scribble entry in the same change —
documentation lives in `torch/scribblings/*.scrbl`, never in a comment above
the definition:

```bash
nix develop .#ci --command \
  raco scribble --dest /tmp/doc torch/scribblings/torch.scrbl
```

### 8. Final gate

```bash
git add -A && nix flake check
```

What CI runs, minus the Resyntax lane (step 5) and the coverage floor
(step 4). Green here means the racket jobs pass in CI.

## Conventions the gates assume

- **Contracts at the definition site**, via `define/contract-out` /
  `define/checked-out` (`torch/private/contract.rkt`), never a
  `contract-out` block in the facade. `torch/generated.rkt` and
  `torch/foreign/raw/` never carry contracts; a name promoted from either is
  contracted at the promotion site with the value form.
- **Validate in the contract, not in the body.** An `->i` with `#:pre/desc`
  gives the caller blame and keeps the body free of `unless`/`error` guards.
  A guard the contract already covers is dead code coverage will flag.
- **Imports** are `(only-in ...)` with explicit alphabetized name lists.
  Exemptions (pure re-export facades, macro-heavy modules) carry a comment at
  the require site.
- **Modules target ≤ 500 lines.** Split by family rather than fighting it.
- **Shadowing racket/base** (`exp log sqrt max min + - * /` …) means the new
  op must dispatch: tensors to libtorch, everything else to the original, so
  `(require torch)` never breaks numeric code. Scribble examples then need
  `(for-label (except-in racket/base ...))`.
- **Comments** explain a constraint the code cannot show. Usage documentation
  belongs in Scribble; narration and review archaeology get deleted.
