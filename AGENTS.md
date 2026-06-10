# AGENTS.md

## Project Overview

`torchrkt` provides Racket bindings to **libtorch** (the C++ core of PyTorch).
**v1** is in: the v0 scaffold (pipeline + handle/finalizer/last-error
substrate, `plans/v0-scaffold.md`) plus the curated tensor-op tranche,
autograd, and the `define-module` nn system — all validated against PyTorch.
Design rationale and the closed nn-architecture decision:
`docs/design/v1-codegen-nn.md`; task ledger: `plans/v1.md`.

The Racket package is the `torchrkt` collection:

- `(require torchrkt)` — the high-level API. A `tensor` is a wrapper struct whose
  native handle is reclaimed by Racket's GC; user code never frees it.
- `(require torchrkt/nn)` — the nn layer (mirrors `import torch.nn`):
  `define-module`, `gen:module`, `linear`, `sgd`, `mse-loss`, initializers.
- `(require torchrkt/foreign)` — the contracted low-level layer. Same surface,
  applied contracts; this file is the authoritative description of the API.
- `(require (submod torchrkt/foreign unsafe))` — adds `tensor-free!` for
  deterministic release. Idempotent: a second free raises `exn:fail:contract`
  at the contract boundary instead of double-freeing.
- `(require torchrkt/foreign/raw/*)` — the direct C FFI layer.

The native bridge is a C++ shared library, `libtorchrkt`, built with CMake and
linked against libtorch via `find_package(Torch)`.

### v1 surface

CPU + float32 only. From `torchrkt`:

- v0 core: `torch-version manual-seed! randn tensor-shape tensor-numel
  tensor->list tensor->vector tensor->repr tensor->string`
- creation: `zeros ones full arange eye tensor rand` (+ in-place `uniform!`)
- shape: `reshape view transpose permute squeeze unsqueeze cat stack`
- elementwise: `add sub mul div pow neg exp log sqrt relu sigmoid tanh`
  (binary ops take a real on either side)
- reductions: `sum mean max min argmax softmax log-softmax`
- linalg: `matmul mm mv dot`; out: `item to-dtype`
- autograd: `requires-grad! requires-grad? backward! grad has-grad? detach
  with-no-grad grad-enabled?`; in-place `sub! zero! mul! zero-grad!`

**Name shadowing convention:** ops colliding with racket/base or racket/list
(`exp log sqrt max min argmax`) are generic — tensors hit libtorch, anything
else defers to the original — so `(require torchrkt)` never breaks numeric
code. New ops that collide must follow the same dispatch pattern, and
scribble examples need `(for-label (except-in racket/base ...))`.

From `torchrkt/nn`: `define-module gen:module module? parameters
named-parameters buffers forward linear sgd step! zero-grads! mse-loss
kaiming-uniform uniform-init fan-in`. `define-module` is the Python-style
`nn.Module` analog: fields are registered at expansion time, models are
plain struct trees owned by the GC (no global parameter store), and
`prop:procedure` makes `(net x)` work like `__call__`. Layer init mirrors
PyTorch RNG consumption (`nn.Linear.reset_parameters`), so a shared
`manual-seed!` yields bit-comparable parameters — the MLP cross-test relies
on this.

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
  tensor, creation, shape_ops, elementwise, reduce, linalg, autograd).
  Integer-status + size-then-fill + `tr_last_error` contract; opaque
  `tr_tensor` handles are returned by constructors/ops and freed by
  `tr_tensor_free`.
- `src/torchrkt/*.cpp` — translation layer; catches C++ exceptions, returns
  status codes / NULL. `detail/tensor_handle.hpp` (in `src/`, private)
  completes the opaque struct over a `torch::Tensor`;
  `detail/op_call.hpp` holds the boundary helpers (`alloc_result`,
  `status_call`, `null_arg`) every op body reduces to — new ops must use
  them rather than hand-rolling try/catch.
- `tests/torchrkt/{random,ops,autograd}_test.cpp` — GoogleTest goldens per
  family. `c_api_compile_test.c` proves the headers are valid C (add a
  function-pointer line per new op family).

### Racket (`torchrkt/`)

Thin re-export facades over small modules (target ≤ 500 lines/file):

- `info.rkt` — package metadata + native-library pre-install hook.
- `main.rkt` — high-level facade (re-exports `foreign.rkt`).
- `foreign.rkt` — the contracted layer + the `unsafe` submodule.
- `foreign/ops.rkt` — version/seed/randn + marshalling (`item`, `to-dtype`,
  `rand`, `uniform!`); `foreign/tensor-ops.rkt` — the op tranche (and the
  shadow-dispatch convention); `foreign/autograd-ops.rkt` — autograd +
  `with-no-grad` + in-place ops; `foreign/structs.rkt` — the `tensor`
  wrapper (`prop:cpointer`, shape cached at wrap time, allocator/deallocator
  finalizer); `foreign/error.rkt` — `check-ok` / `check-handle`;
  `foreign/format.rkt` — the PyTorch-repr reproducer.
- `foreign/raw/*.rkt` — direct FFI, one module per C translation unit:
  `library` (the definer), `global`, `tensor` (`_Tensor` cpointer +
  deallocator), `random`, `creation`, `shape-ops`, `elementwise`, `reduce`,
  `linalg`, `autograd`. Tensor-returning bindings always carry
  `#:wrap (allocator tr-tensor-free/raw)`.
- `nn.rkt` — contracted facade over `nn/` (`module.rkt` = `gen:module` +
  the `define-module` macro; `linear.rkt`, `init.rkt`, `optim.rkt`,
  `loss.rkt`).
- `private/install-torchrkt-native.rkt` — stages `libtorchrkt.*` into
  `native-libs/` from `TORCHRKT_NATIVE_LIB_PATH` (set by the Nix build/shell).
  NOTE: the dev shell only re-stages on first provision (the `deps_stamp`
  guard); after changing C++, re-copy from `nix build .#cpp
  --print-out-paths` (or `nix run .#copy-native-libs`) before `raco test`.

## CI

`.github/workflows/nix.yml`: `nix flake check` on `ubuntu-latest` +
`macos-latest`, plus a `resyntax` lint gate. (A `raco-catalog` workflow is
deferred until the portable native-candidate story exists — libtorch is too
large to bundle the xgboost way.)
