#!/usr/bin/env bash
#
# Run clang-tidy over the given TUs in parallel, one process per file. Backs the
# cmake `tidy` target. Parallelism = $NIX_BUILD_CORES (set by `nix build
# --cores`, which we cap at 6 on the lab host for thermal reasons, and which CI
# sets to its core count); defaults to 4 for a bare `cmake --build` in a dev
# shell. clang-tidy is single-threaded per file, so N processes ≈ N cores.
#
# Usage: tidy-parallel.sh <clang-tidy> <build-dir> <file>...
set -euo pipefail

clang_tidy="$1"
build="$2"
shift 2

jobs="${NIX_BUILD_CORES:-4}"
[ "$jobs" -ge 1 ] 2>/dev/null || jobs=4

printf '%s\n' "$@" \
  | xargs -r -P "$jobs" -I{} "$clang_tidy" --quiet -p "$build" --extra-arg=-w {}
