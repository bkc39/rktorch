# rktorch v0 — scaffold a libtorch binding with SOTA dev-ops

## Context

This recovers and concretizes the planning session that was interrupted by a hard reboot
(`torch-plan.md` holds the original brain-dump; this is the executable version).

Goal of v0 is **not** to build out a torch API. It is to **stand up the repository — `rktorch`
— with the same SOTA build/test/lint/CI pipeline as `xgboost-rkt`, proven end-to-end by the
thinnest possible vertical slice through the whole stack**: Nix-built C++ shim → Racket FFI →
example → PyTorch parity check → green CI on Linux + macOS. Everything hard about torch
(codegen from `native_functions.yaml`, `nn.Module` macros, the full ATen surface, CUDA, MPS)
is **explicitly deferred**. We are building the *skeleton and the spinal cord*, one nerve.

The template is already in this tree and is mature; we mirror it rather than invent:
- **`xgboost-rkt/`** — the ideal C++→Racket FFI template (opaque handles + GC finalizers, thin
  `extern "C"` shim, `last_error` propagation, Nix-staged `native-libs/`, line-count + format +
  tidy gates, 4-platform flake, CI matrix, Resyntax/review linters).
- **`scs/scs/tests/python-cross-test.rkt`** — the model for the PyTorch-parity harness
  (run adjacent Python file → JSON → compare to Racket with tolerance → **SKIP** if the Python
  lib can't import, so `raco test` / `nix build` stay green without the dev shell).
- **`torch/`** (old) — a dead stub (`test()` returns `42`, broken `tensor` struct). **Do not reuse.**

## Decisions (locked for v0, open to iteration)

| Question | Decision | Why |
|---|---|---|
| libtorch source | **`pkgs.libtorch-bin` 2.9.0** + **tolerant** Python cross-test | Tiny prebuilt download → fast, cache-friendly CI on both target platforms; covers darwin+linux. |
| v0 surface | **Smallest possible**: `manual-seed!` + `randn` (fixed 2×2) + tensor `print` + the parity example. | One handle-returning fn + one effectful fn exercises the entire FFI lifetime/finalizer path. |
| Platforms | **`aarch64-darwin` + `x86_64-linux`** only | Exactly what `libtorch-bin` ships (verified via `nix eval`). Drop the other two flake systems for now. |
| MPS / darwin-CPU | **Deferred**; CPU device only in v0 | v0 ops are CPU-only anyway. See note below — MPS likely comes *free* from the prebuilt. |

## Your question, answered: "build against python torch? rebuild pytorch on our libtorch?"

Two things to untangle, because the nixpkgs facts (verified by `nix eval` against
nixpkgs `26.11.20260606`) reframe it:

1. **"Rebuild pytorch using our nix-managed libtorch" isn't a real layering.** In PyTorch's
   own build, **libtorch *is* the C++ core of pytorch** — `libtorch_cpu.{so,dylib}` etc. are
   produced by the *same* source build that produces the Python bindings. `libtorch-bin` is
   just that C++ core, pre-extracted, with **no matching Python interpreter**. You can't bolt
   the Python bindings onto a standalone `libtorch-bin`; they're one tree, not two layers.

2. **The inverse is exactly what you want, and Nix gives it for free.**
   `python3Packages.torch` (2.12.0, built *from source* in nixpkgs) exposes
   `dev` / `lib` / `cxxdev` outputs — i.e. **a complete, `find_package(Torch)`-discoverable
   libtorch lives *inside* the same derivation the parity script does `import torch` from.**
   So "build against python torch" = point our shim's `find_package(Torch CONFIG REQUIRED)` at
   `${python3Packages.torch.dev}/.../torch/share/cmake/Torch`. Then C++ and Python share **one
   ATen build** → seeded `randn` is **bit-identical** with zero version drift. **No rebuild,
   just a different `buildInputs` + `Torch_DIR`.**

**Recommendation for v0: still `libtorch-bin`.** Rationale: (a) the v0 goal is "get the
pipeline green fast," and `libtorch-bin` is a small cached download on both target platforms,
whereas source `python3Packages.torch` on **`aarch64-darwin` is unlikely to be in the binary
cache → CI compiles torch for hours / times out**; (b) seeded CPU `randn` is in practice stable
across 2.9↔2.12 (same MT19937 generator + normal transform), so the tolerant test will very
likely pass tightly anyway. We bake the **bit-exact upgrade in as a one-flag switch** (a
`torchSource = "bin" | "python"` knob in the flake) so we can flip to guaranteed parity the
day we want it and accept the heavier build. This directly preserves the parity goal without
paying its cost up front.

> MPS note: upstream `libtorch` arm64-macOS binaries ship with the MPS backend compiled in, so
> `libtorch-bin` on `aarch64-darwin` *probably* already gives us MPS with no source build — to
> be **verified at build time**, not assumed. A separate "darwin CPU build" is not a separate
> libtorch; it's just selecting the CPU device at runtime. Both are deferred past v0.

## Repository layout (`rktorch`, mirroring `xgboost-rkt`)

```
rktorch/
  flake.nix                       # systems = [aarch64-darwin x86_64-linux]; torchSource knob
  flake.lock
  AGENTS.md  CLAUDE.md            # CLAUDE.md → "see AGENTS.md"
  .clang-format  .gitignore
  plans/                          # carry this plan + a TODO.md in-repo
  cpp/
    CMakeLists.txt                # find_package(Torch CONFIG REQUIRED); SHARED lib libtorchrkt
    CMakePresets.json  .clang-tidy  .clang-format
    cmake/{Warnings,Tools}.cmake  # warnings + format-check/tidy targets (port from xgbcompat)
    include/torchrkt/c_api.h      # umbrella -> c_api/{tensor,random,global}.h
    include/torchrkt/c_api/{global,random,tensor}.h   # extern "C", size-then-fill, last_error
    src/torchrkt/{global,random,tensor,detail/error}.cpp
    tests/torchrkt/random_test.cpp                    # gtest: seed 0 -> randn 2x2 golden
  torchrkt/
    info.rkt                      # pkg metadata + native-lib pre-install hook
    main.rkt                      # high-level facade (re-export) — #|review: ignore|#
    foreign.rkt                   # safe contracted facade — #|review: ignore|#
    foreign/{error,structs}.rkt   # tensor wrapper struct: prop:cpointer + allocator/deallocator finalizer
    foreign/raw/{library,random,tensor}.rkt           # define-runtime-path "../../native-libs"
    native-libs/                  # Nix stages libtorchrkt.{so,dylib} here (gitignored)
  examples/
    python/00_randn.py            # torch.manual_seed(0); print(json) of randn(2,2)
    racket/00-randn.rkt           # Racket equivalent via (require rktorch)
    test/00-randn.rkt             # runner + RackUnit, mirrors examples/test pattern
  torchrkt/tests/python-cross-test.rkt   # tolerant compare vs 00_randn.py; SKIP if `import torch` fails
  .github/workflows/{nix,raco-catalog,docs,pkg-build-harness}.yml
```

## The vertical slice (what actually gets implemented in v0)

**C++ shim** (`cpp/`, mirrors `xgbcompat`'s `extern "C"` + integer-status + `last_error` contract):
- `tr_last_error(void) -> const char*`
- `tr_manual_seed(uint64_t seed) -> int`
- `tr_randn2x2(float* out4) -> int` (v0 fixes shape 2×2; CPU; writes row-major)
- `tr_tensor_print(...)` via ATen `operator<<` into the size-then-fill buffer contract
- handle type only if we keep a `Tensor*` opaque handle; for "smallest possible" we can return
  the 4 floats directly and defer the opaque-handle struct to the next slice. **Iteration point.**
- `cpp/tests/.../random_test.cpp`: gtest asserting seed-0 `randn(2,2)` equals a captured golden.

**Racket FFI** (`torchrkt/`, mirrors `xgboost-rkt/xgboost/foreign/*`):
- `foreign/raw/library.rkt`: `(define-runtime-path native-libs-dir "../../native-libs")` +
  `(ffi-lib (build-path native-libs-dir "libtorchrkt"))` — identical loader pattern.
- raw bindings for the three C fns; `check-ok` error helper reading `tr_last_error`.
- `foreign.rkt` / `main.rkt`: contracted facade exposing `manual-seed!`, `randn`, tensor print.
- (Tensor wrapper struct with `prop:cpointer 0` + `#:wrap (allocator … (deallocator))` finalizer
  comes online when we introduce the opaque handle — sketch it, wire it when randn returns a handle.)

**Parity** (`torchrkt/tests/python-cross-test.rkt`, mirrors `scs`):
- seed 0 → `randn(2,2)`; run `examples/python/00_randn.py`, read its JSON, compare elementwise
  with tolerance; **SKIP with a printed notice when `python3 -c "import torch"` fails** so plain
  `raco test` / `nix build` stay green outside `nix develop`.

## Dev-ops suite to port verbatim (the actual point of v0)

- **`flake.nix` outputs**: `cpp` (CMake build + `ctest` via `doCheck`), `cpp-format`, `cpp-tidy`,
  `cpp-line-count` (500-line/file gate), `racket` (installs pkg, stages `native-libs/`, runs
  `raco test torchrkt/` + `raco test examples/test/`), `copy-native-libs` app, and a `default`
  dev shell that provisions **Resyntax + racket-review** into user scope. Add the
  `torchSource` knob selecting `buildInputs = [ libtorch-bin ]` vs `[ python3Packages.torch{,.dev} ]`
  with the matching `Torch_DIR`. Drop the glibc-polyfill / aarch64-cross / CUDA branches for v0.
- **CI** (`.github/workflows/`): `nix.yml` = `nix flake check` matrix on `ubuntu-latest` +
  `macos-latest`, plus the `resyntax` gate job. Port `raco-catalog.yml`, `docs.yml`,
  `pkg-build-harness.yml` as-is (trim to the v0 surface).
- **Conventions** (`AGENTS.md`): thin re-export facades carry `#|review: ignore|#`; ≤500
  lines/file; high-level `(require rktorch)` is GC-managed (no manual free); `foreign` is the
  contracted layer; `foreign/raw` is direct FFI.

## Verification (end-to-end, CI-equivalent)

```bash
cd rktorch
nix build .#cpp                       # CMake + gtest golden (seed-0 randn 2x2)
nix build                             # builds shim, installs pkg, runs raco test + examples
nix develop --command bash -c '
  raco test torchrkt/                 # FFI unit tests
  raco test examples/test/            # 00-randn runner
  raco test torchrkt/tests/python-cross-test.rkt   # real parity vs python torch (in-shell)
  resyntax analyze --directory torchrkt            # lint gate (must report no suggestions)
'
racket examples/racket/00-randn.rkt   # prints the 2x2 tensor
```
Done when: `nix flake check` is green on both `aarch64-darwin` and `x86_64-linux`, the parity
test passes in-shell and SKIPs cleanly out-of-shell, and Resyntax is clean.

## Open questions to iterate on (before/while building)

1. **Opaque tensor handle in v0, or defer?** Returning 4 floats is the truly-smallest slice but
   doesn't exercise the GC-finalizer path (the most template-valuable bit). Recommend: introduce
   the `_tensor` opaque handle + finalizer now, with `randn` returning a handle and a
   `tensor->list` extractor — still tiny, but proves the lifetime model. (This is the "a bit more"
   shading toward your "smallest" — your call.)
2. **`torchSource` default**: ship `bin` (this plan) and keep `python` one flag away — agreed?
3. **Repo location / name**: confirm `rktorch` and whether it's a sibling dir here +
   new GitHub repo `bkc39/rktorch` mirroring the others.
4. **`native_functions.yaml` codegen**: confirmed *out of scope* for v0 — we hand-write 3 fns.
   The codegen-vs-runtime-macro design question from `torch-plan.md` is revisited at v1.
