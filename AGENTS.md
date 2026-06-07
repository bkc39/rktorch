# AGENTS.md

## Project Overview

`torchrkt` provides Racket bindings to **libtorch** (the C++ core of PyTorch).
This is the **v0 scaffold**: a thin vertical slice — seed the RNG, draw a tensor,
read it back, and check it against PyTorch — wired through the full
build/test/lint/CI pipeline. The point of v0 is the *pipeline*, not the API
surface. See `plans/v0-scaffold.md` for scope and the decisions behind it.

The Racket package is the `torchrkt` collection:

- `(require torchrkt)` — the high-level API. A `tensor` is a wrapper struct whose
  native handle is reclaimed by Racket's GC; user code never frees it.
- `(require torchrkt/foreign)` — the contracted low-level layer. Same surface,
  applied contracts; this file is the authoritative description of the API.
- `(require (submod torchrkt/foreign unsafe))` — adds `tensor-free!` for
  deterministic release. Idempotent: a second free raises `exn:fail:contract`
  at the contract boundary instead of double-freeing.
- `(require torchrkt/foreign/raw/*)` — the direct C FFI layer.

The native bridge is a C++ shared library, `libtorchrkt`, built with CMake and
linked against libtorch via `find_package(Torch)`.

### v0 surface

`torch-version`, `manual-seed!`, `randn` (arbitrary shape, returns a tensor
handle), `tensor-shape`, `tensor-numel`, `tensor->list`, `tensor->vector`,
`tensor->repr`, `tensor->string`. CPU + float32 only.

**REPL parity.** A tensor prints in the Racket REPL exactly as it does in the
Python REPL — `tensor([[ 1.5410, -0.2934], [-2.1788, 0.5684]])` — via
`prop:custom-write`, reproduced from the data + shape in `foreign/format.rkt`
(PyTorch's `tensor(...)` framing isn't in libtorch's C++ printer). Two
accessors expose the two forms explicitly: `tensor->repr` is the PyTorch repr
(what the REPL shows); `tensor->string` is ATen's C++ `operator<<` text. The
repr reproducer covers the common case (CPU float32, fixed-point, precision 4)
and falls back to the ATen form for scientific-notation values; see the TODO in
`foreign/format.rkt`.

Deferred (see plan): `native_functions.yaml` codegen, `nn.Module` macros, the
broader ATen surface, CUDA, MPS, and the portable raco-catalog candidate story.

## The libtorch source knob

`flake.nix` has `torchSource = "bin" | "python"`:

- **`bin`** (default) — `pkgs.libtorch-bin`. Small prebuilt download, fast cached
  CI on `aarch64-darwin` + `x86_64-linux`. Parity with Python torch is *tolerant*
  (the cross-test uses a float tolerance), because the C++ side may be a
  different patch version than the Python torch.
- **`python`** — `pkgs.python3Packages.torch`. Builds against the *same* libtorch
  the parity script imports, so seeded `randn` is **bit-exact** — at the cost of
  a heavy (often uncached on darwin) from-source build.

## Build Commands

```bash
nix build              # builds cpp, installs the pkg, runs raco test + examples
nix build .#cpp        # CMake build + gtest only
nix flake check        # everything: cpp, format, tidy, line-count, racket
nix develop            # dev shell (includes a Python with `torch`)
nix develop .#ci       # lean shell without Python torch (used by the lint job)
./result/bin/torchrkt  # runs (module+ main): prints version + a 2x2 draw
```

Inside `nix develop`:

```bash
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure

raco test torchrkt/          # FFI unit tests (+ self-skipping parity test)
raco test examples/test/     # literate-example runners
racket -l torchrkt           # REPL with the package

resyntax analyze --directory torchrkt   # lint gate (CI fails on any suggestion)
resyntax fix --directory torchrkt
raco review torchrkt/**/*.rkt
```

`raco review` does not expand macros, so the pure re-export facades
(`main.rkt`, `foreign.rkt`) and `info.rkt` carry a `#|review: ignore|#` directive.

### PyTorch parity

The default `nix develop` shell ships a Python with `torch`, so you can explore
PyTorch behaviour beside the Racket bindings and run the real cross-test:

```bash
nix develop --command python3 -c 'import torch; print(torch.__version__)'
nix develop --command raco test torchrkt/tests/python-cross-test.rkt
```

Where python3 can't `import torch` (the sandboxed `nix build`, or the lean
`.#ci` shell), the test self-skips, so `nix build` / `raco test` stay green.

## Architecture

### C++ (`cpp/`)

- `include/torchrkt/c_api/*.h` — the `extern "C"` FFI surface (global, random,
  tensor). Integer-status + size-then-fill + `tr_last_error` contract; the
  opaque `tr_tensor` handle is returned by `tr_randn` and freed by
  `tr_tensor_free`.
- `src/torchrkt/*.cpp` — translation layer; catches C++ exceptions, returns
  status codes / NULL. `detail/tensor_handle.hpp` (in `src/`, private) completes
  the opaque struct over a `torch::Tensor`.
- `tests/torchrkt/random_test.cpp` — GoogleTest: shape, determinism (same seed →
  same draws), the size-then-fill probes. `c_api_compile_test.c` proves the
  headers are valid C.

### Racket (`torchrkt/`)

Thin re-export facades over small modules (target ≤ 500 lines/file):

- `info.rkt` — package metadata + native-library pre-install hook.
- `main.rkt` — high-level facade (re-exports `foreign.rkt`).
- `foreign.rkt` — the contracted layer + the `unsafe` submodule.
- `foreign/ops.rkt` — safe operations; `foreign/structs.rkt` — the `tensor`
  wrapper (`prop:cpointer`, shape cached at wrap time, allocator/deallocator
  finalizer); `foreign/error.rkt` — `check-ok` / `check-handle`.
- `foreign/raw/*.rkt` — direct FFI: `library` (the definer), `global`, `tensor`
  (`_Tensor` cpointer + deallocator), `random` (`tr-randn` + allocator).
- `private/install-torchrkt-native.rkt` — stages `libtorchrkt.*` into
  `native-libs/` from `TORCHRKT_NATIVE_LIB_PATH` (set by the Nix build/shell).

## CI

`.github/workflows/nix.yml`: `nix flake check` on `ubuntu-latest` +
`macos-latest`, plus a `resyntax` lint gate. (A `raco-catalog` workflow is
deferred until the portable native-candidate story exists — libtorch is too
large to bundle the xgboost way.)
