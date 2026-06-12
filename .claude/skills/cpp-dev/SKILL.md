---
name: cpp-dev
description: Local C++ development loop for the torch shim — format, lint, line-gate, build, and test every C++ change before calling it done. Use this whenever editing, adding, or reviewing anything under cpp/ (.cpp/.h/.hpp files, CMakeLists.txt, new ATen op bindings, gtest files), and whenever a C++ change needs to be picked up by the Racket tests. Also use it when CI fails on cpp-format, cpp-tidy, or cpp-line-count and the fix needs to be reproduced locally.
---

# C++ development loop (torch)

All the gates that CI runs (`nix flake check`: `cpp`, `cpp-format`, `cpp-tidy`,
`cpp-line-count`) are runnable locally, and running them *during* development
is much cheaper than discovering failures at the end. This skill is the
ordered loop a C++ change goes through. The tooling was ported from the
sibling `xgboost-rkt` repo and is configured in-repo: `cpp/.clang-format`,
`cpp/.clang-tidy`, the `format-check`/`tidy` CMake targets
(`cpp/cmake/TorchrktTools.cmake`), and the 500-line gate in `flake.nix`.

## Before hand-writing a new op: is it allowlist-eligible?

The codegen generator (`nix run .#codegen`, see
`codegen/` and the AGENTS.md codegen section) emits the whole three-layer
stack — `extern "C"` shim, raw Racket binding, uncontracted wrapper — for
any ATen op whose signature fits the IR (Tensor / Scalar→double / int64 /
bool / IntArrayRef / TensorList args, single Tensor return). **If the op is
IR-eligible, add a line to `codegen/allowlist.txt` + an input recipe in
`torch/tests/python-cross-test.rkt`, regenerate, and skip hand-writing
entirely.** Hand-write only ops outside the IR (multi-return, out-params,
optional args, dtype/device knobs). Never edit files under a `generated/`
path or `torch/generated.rkt` — CI's `codegen-drift` job fails on any
divergence from the generator's output.

## The loop

### 0. Stage before any nix command

The flake builds from the **git-tracked tree**: untracked files are invisible
to `nix develop` / `nix build` / `nix flake check`, producing confusing
"Cannot find source file" CMake errors. After creating or editing files:

```bash
git add -A   # or add the specific new files
```

### 1. Format as you go (don't batch it for the end)

After writing or editing any `.c/.cpp/.h/.hpp` file:

```bash
nix develop --command bash -c 'cd cpp && clang-format -i \
  include/torchrkt/c_api/*.h src/torchrkt/*.cpp src/torchrkt/detail/*.hpp \
  tests/torch/*.cpp tests/torch/*.c'
```

clang-format reflows lambda-heavy code (the `alloc_result`/`status_call`
boundary calls) in ways that are hard to hand-predict, so write naturally and
let the tool own the layout. Verify with `--dry-run --Werror` on the same file
list if unsure.

### 2. Line gate: 500 lines per file

`flake.nix` fails any C++ file over 500 lines. When a translation unit
approaches the limit, split by op family (the `creation/shape_ops/elementwise/
reduce/linalg` pattern) rather than fighting the gate. Quick check:

```bash
find cpp/include cpp/src cpp/tests -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.c' | xargs wc -l | sort -n | tail -5
```

### 3. Build + gtest

```bash
nix develop --command bash -c \
  'cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON \
   && cmake --build cpp/build \
   && ctest --test-dir cpp/build --output-on-failure'
```

Every new op family gets gtest goldens in `cpp/tests/torchrkt/` (mirror
`ops_test.cpp`: the RAII `Handle` wrapper, `data_of`/`shape_of` probes, and
one error-path test per family — NULL/shape-mismatch must surface as status
codes, never aborts). Watch for C++ keyword collisions in test code
(`requires` is reserved in C++20).

### 4. clang-tidy

```bash
nix develop --command bash -c 'cmake --build cpp/build --target tidy'
```

Checks enabled: `bugprone-*`, `performance-*`, `readability-*` (cognitive
complexity ≤ 40), `modernize-use-nullptr/override`. New op bodies stay tidy-
clean automatically if they reduce to the shared boundary helpers in
`cpp/src/torchrkt/detail/op_call.hpp` (`alloc_result`, `status_call`,
`null_arg`) instead of hand-rolled try/catch.

### 5. New public C functions: three sync points

Adding to the `extern "C"` surface requires updating, in the same change:

1. `cpp/CMakeLists.txt` — add the new `.cpp` to `target_sources` (and any new
   test file to `torchrkt_tests`)
2. `cpp/include/torchrkt/c_api.h` — include the new header (alphabetical)
3. `cpp/tests/torchrkt/c_api_compile_test.c` — one function-pointer line per
   new function/family, proving C linkage

### 6. Re-stage the native lib before Racket tests

The dev shell only copies `libtorchrkt` into `torch/native-libs/` on first
provision (the `deps_stamp` guard), so Racket tests silently run against the
**stale** library after C++ changes — symptom: `dlsym ... symbol not found`.
After any C++ change that Racket code will exercise:

```bash
CPP_OUT=$(nix build .#cpp --print-out-paths | tail -1) \
  && rm -f torch/native-libs/libtorchrkt.* \
  && cp "$CPP_OUT"/lib/libtorchrkt.* torch/native-libs/ \
  && chmod u+w torch/native-libs/libtorchrkt.*
```

Then `nix develop --command raco test torch/`. If Racket modules fail with
"reference to a variable that is not exported" or stale-binding errors, clear
the project's `compiled/` dirs (`find torch examples -type d -name
compiled -exec rm -rf {} +`) and rerun.

### 7. Final gate

Before declaring the change done:

```bash
git add -A && nix flake check
```

This is exactly what CI runs (plus the resyntax job for Racket files). If it
passes locally, the cpp jobs pass in CI.

## Conventions the gates assume

- Every boundary function catches all exceptions and reports via
  `torchrkt::set_error` + integer status / NULL return — nothing throws across
  the FFI line. Use the `op_call.hpp` helpers; don't hand-roll.
- Tensor-returning C functions follow `tr_<op>(...) -> tr_tensor*` (NULL on
  error) so the Racket side can wrap them with
  `#:wrap (allocator tr-tensor-free/raw)` uniformly.
- Size-then-fill probes return rc=2 with the required size in the out-param.
- Comment style: explain the constraint the code can't show (see existing
  files); match 2-space indent, 80-column format.
