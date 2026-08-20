# AGENTS.md

## Project Overview

`torch` provides Racket bindings to **libtorch** (the C++ core of PyTorch).
**v1** is in: the v0 scaffold (pipeline + handle/finalizer/last-error
substrate, `plans/v0-scaffold.md`) plus the curated tensor-op tranche,
autograd, and the `define-module` nn system — all validated against PyTorch.
Design rationale and the closed nn-architecture decision:
`docs/design/v1-codegen-nn.md`.

The Racket package is the `torch` collection:

- `(require torch)` — the high-level API. A `tensor` is a wrapper struct whose
  native handle is reclaimed by Racket's GC; user code never frees it.
- `(require torch/nn)` — the nn layer (mirrors `import torch.nn`):
  `define-module`, `gen:module`, `Linear`, `sgd`, `mse-loss`, initializers.
  **Naming convention:** nn layer *constructors* are PascalCase (`Linear`,
  `Conv2d`, `MaxPool2d`, `Flatten`, `Dropout`, `Sequential`, `Embedding`,
  `LayerNorm`), mirroring the `torch.nn.*` classes; their *predicates* are
  lowercase (`linear?`, `conv2d?`, `max-pool2d?`, `flatten?`, `dropout?`,
  `sequential?`, `embedding?`, `layer-norm?`), per Racket idiom
  (`list?`, `hash?`). The functional ops keep lowercase names on `torch`
  (`conv2d`, `max-pool2d`, `flatten`, like `torch.conv2d`). The PascalCase
  constructors vs lowercase functional ops are what let `(require torch
  torch/nn)` coexist without collision (#11).
- `(require torch/foreign)` — the contracted low-level layer. Same surface,
  applied contracts; this file is the authoritative description of the API.
- `(require (submod torch/foreign unsafe))` — adds `tensor-free!` for
  deterministic release. Idempotent: a second free raises `exn:fail:contract`
  at the contract boundary instead of double-freeing.
- `(require torch/foreign/raw/*)` — the direct C FFI layer.

The native bridge is a C++ shared library, `libtorchrkt`, built with CMake and
linked against libtorch via `find_package(Torch)`.

### v1 surface

CPU + float32 only. From `torch`:

- v0 core: `torch-version manual-seed! randn tensor-shape tensor-numel
  tensor->list tensor->vector tensor->repr tensor->string`
- memory: `native-memory-use` (per-device outstanding native bytes from
  the #37 ledger), `tensor-free!` (explicit synchronous release)
- creation: `zeros ones full arange eye tensor rand` (+ in-place `uniform!`)
- shape: `reshape view transpose permute squeeze unsqueeze cat stack`
- elementwise: `add sub mul div pow neg exp log sqrt relu sigmoid tanh`
  (binary ops take a real on either side)
- operators: `+ - * /` shadow racket/base rkt-polars-style (numeric fast
  path to racket/base, tensor operands dispatch to add/sub/mul/div, chains
  fold left); `@` is matmul, like Python's `a @ b`; `t`/`Σ` are terse
  aliases for transpose/sum; `~> ~>> lambda~> lambda~>>` are re-provided
  from the `threading` library (dep `threading-lib`, prefetched offline by
  the `racket-deps` fixed-output derivation in flake.nix)
- reductions: `sum mean max min argmax softmax log-softmax`
- linalg: `matmul mm mv dot`; out: `item to-dtype`
- autograd: `requires-grad! requires-grad? backward! grad has-grad?
  maybe-grad detach with-no-grad grad-enabled?`; in-place
  `sub! zero! mul! zero-grad!`

**Name shadowing convention:** ops colliding with racket/base or racket/list
(`exp log sqrt tanh max min argmax`) are generic — tensors hit libtorch,
anything else defers to the original — so `(require torch)` never breaks
numeric code. New ops that collide must follow the same dispatch pattern
(check racket/base first — `tanh` was missed initially and broke numeric
callers), and scribble examples need
`(for-label (except-in racket/base exp log sqrt max min + - * /))`.
Dispatching named ops carry dependent (`->i`) contracts so the wrong shape
gets contract blame, not a runtime error; the `+ - * / @` operators are
provided as plain renames (no contract overhead on the numeric fast path),
per `foreign/operators.rkt`.

From `torch/nn`: `define-module gen:module module? parameters
named-parameters buffers forward Linear Conv2d MaxPool2d Flatten Dropout
Sequential Embedding LayerNorm sgd adam step! zero-grads! cross-entropy
mse-loss kaiming-uniform uniform-init normal-init fan-in`. The functional
transformer primitives (`gelu tril triu masked-fill embedding layer-norm`,
tranche 3, #22) live on `torch` beside the other functional ops; the GPT
causal-mask idiom is `(masked-fill scores (eq (tril (ones T T)) 0) -inf.0)`. `define-module` is the Python-style
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
./result/bin/torch  # runs (module+ main): prints version + a 2x2 draw
```

Inside `nix develop`:

```bash
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure

raco test torch/          # FFI unit tests (+ self-skipping parity test)
raco test examples/test/     # literate-example runners
racket -l torch           # REPL with the package

resyntax analyze --directory torch   # lint gate (CI fails on any suggestion)
resyntax fix --directory torch
raco review torch/**/*.rkt
```

`raco review` does not expand macros, so the pure re-export facades
(`main.rkt`, `foreign.rkt`) and `info.rkt` carry a `#|review: ignore|#` directive.

### PyTorch parity

The default `nix develop` shell ships a Python with `torch`, so you can explore
PyTorch behaviour beside the Racket bindings and run the real cross-test:

```bash
nix develop --command python3 -c 'import torch; print(torch.__version__)'
nix develop --command raco test torch/tests/python-cross-test.rkt
nix develop --command raco test torch/tests/generated-parity-test.rkt
```

Where python3 can't `import torch` (the sandboxed `nix build`, or the lean
`.#ci` shell), the tests self-skip, so `nix build` / `raco test` stay green.

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
  `detail/op_call.hpp` holds the boundary helpers (`alloc_result` and
  `null_arg` for tensor-returning ops; `status_call` and `null_arg_status`
  for the int-status in-place shape) every op body reduces to — new ops
  must use them rather than hand-rolling try/catch.
- `tests/torchrkt/{random,ops,autograd,generated_golden,generated_tranche2}_test.cpp`
  — GoogleTest goldens per family (generated families get a C-boundary
  golden: a correctness case + a null/length-guard case).
  `c_api_compile_test.c` proves the headers are valid C (add a
  function-pointer line for at least one representative of each new op
  family, plus any function whose signature shape is new).

### Racket (`torch/`)

Thin re-export facades over small modules (target ≤ 500 lines/file).

**Import convention:** library modules use `(require (only-in ...))` with
explicit, alphabetized name lists — never whole-module requires — so each
file documents exactly what it pulls in. Exemptions, each marked with a
comment at the require site: pure re-export facades (`main.rkt`,
`foreign.rkt`, `nn.rkt`), and macro-heavy modules whose expansions need the
module's full export set (`racket/runtime-path`, `syntax/parse/pre`).

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
  `syntax` (the pure FFI definer + `_Tensor` cpointer), `memory` (the
  lifetime/#37 substrate: frees, pressure ledger, `tensor-allocator`,
  op-definer macros), `global`, `tensor`, `random`, `creation`,
  `shape-ops`, `elementwise`, `reduce`, `linalg`, `autograd`. Tensor-returning bindings always carry
  `#:wrap tensor-allocator` (see `raw/memory.rkt`), which composes the
  finalizer registration (`allocator` over the guarded, finalizer-context
  `tr-tensor-free/finalizer`) with the #37 memory-pressure ledger charge
  (phantom bytes + per-device accounting) and the #38 OOM
  collect-and-retry (one GC + finalizer drain, then exactly one retry,
  keyed off `tr_last_error_kind`). Bindings that draw from the global
  RNG stream (randn/rand; generated ops flagged `rng` in the codegen
  allowlist, e.g. dropout) take `tensor-allocator/rng` instead — the
  same wrap minus the retry, so a retry can never double-draw and break
  seeded parity. OOM-classified failures reach users as
  `exn:fail:rktorch:oom` (catch by type, not message). Never hand-write
  a bare `(allocator ...)` wrap — it would skip the ledger. Explicit synchronous
  release goes through the raising, finalizer-cancelling
  `tr-tensor-free/checked`.
- `nn.rkt` — contracted facade over `nn/` (`module.rkt` = `gen:module` +
  the `define-module` macro; `linear.rkt`, `init.rkt`, `optim.rkt`,
  `loss.rkt`).
- `private/install-torchrkt-native.rkt` — stages `libtorchrkt.*` into
  `native-libs/` from `TORCHRKT_NATIVE_LIB_PATH` (set by the Nix build/shell).
  NOTE: the dev shell only re-stages on first provision (the `deps_stamp`
  guard); after changing C++, re-copy from `nix build .#cpp
  --print-out-paths` (or `nix run .#copy-native-libs`) before `raco test`.

### Codegen (`codegen/`)

The ATen generator (v2/A, #2): `nix run .#codegen` (equivalently
`nix develop --command python3 -m codegen`, but with a much smaller
closure) reads `codegen/allowlist.txt` against the **vendored** schema in
`codegen/aten/` (pinned to the C++ libtorch 2.9.0 — see the README there;
never the dev-shell python torch's copy) and emits, with DO-NOT-EDIT
headers:

- `cpp/{include/torchrkt/c_api,src/torchrkt}/generated/<shard>.{h,cpp}` —
  bodies reduce to the `op_call.hpp` helpers; clang-format is run by the
  generator; `generated/sources.cmake` is included from `cpp/CMakeLists.txt`
- `torch/generated.rkt` — the UNSTABLE uncontracted surface: one compact
  `define-generated-op` form per allowlist entry. The hand-written macro
  in `torch/foreign/define-generated.rkt` owns the expansion into raw FFI
  binding + wrapper, so Racket marshalling knowledge lives in Racket, not
  in Python string templates. Promotion into `torch/foreign.rkt` is
  hand-curated.
- `torch/tests/generated-parity.rktd` — manifest driving the generated-op
  battery in `generated-parity-test.rkt`; every new allowlist line needs an
  input recipe in that test

Conventions:

- **Extend the allowlist instead of hand-writing** when an op fits the IR
  (Tensor / Scalar→double / int64 / bool / IntArrayRef / TensorList args,
  single Tensor return). Unsupported signatures are skipped with a report —
  widening the IR is a generator change, not a hand-written shim.
- Optional *types* (`Tensor?`, `int?`) are outside the IR and skip; schema
  *defaults* (`int dim=0`) are flattened to required arguments on the
  unstable surface — defaults are a curated-facade concern. In-place ops
  (`add_`) skip too: their C-side mutation convention is a #3 decision.
- Generated output is committed (AOT); CI's `codegen-drift` job regenerates
  and fails on any diff, so never edit generated files by hand.
- `generated/` is exempt from the C++ 500-line gate (shard size is the
  generator's concern).
- The golden-equivalence proof lives in
  `cpp/tests/torchrkt/generated_golden_test.cpp`: the generated linalg four
  (`tr_gen_{matmul,mm,mv,dot}`) stay permanently allowlisted and bit-checked
  against the authoritative hand-written family.

## CI

`.github/workflows/nix.yml`: `nix flake check` on `ubuntu-latest` +
`macos-latest`, a `resyntax` lint gate, and a `codegen-drift` job
(regenerate + fail on porcelain). (A `raco-catalog` workflow is deferred
until the portable native-candidate story exists — libtorch is too large to
bundle the xgboost way.)
