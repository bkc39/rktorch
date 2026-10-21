#!/usr/bin/env bash
#
#   scripts/platform/cuda-spike/store-paths.sh   (prints the .#cuda inputs as JSON)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
cd "$(git -C "$here" rev-parse --show-toplevel)"
nix --max-jobs 1 --cores 4 eval --impure --json --apply "import $here/store-paths.nix" \
  .#packages.x86_64-linux.cpp-cuda
