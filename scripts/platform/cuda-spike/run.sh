#!/usr/bin/env bash
#
#   scripts/platform/cuda-spike/run.sh <work-dir>
#
# The K0 spike (#359), end to end: stock Racket 9.1, the three CUDA
# packages and their sizes, a directory catalog, and the install and
# 'cuda training run outside Nix. Needs Nix (to read and patch the .#cuda
# store paths), jq, an NVIDIA driver for CUDA 13, and about 12 GB of disk.
set -euo pipefail

work=$(realpath -m "${1:?usage: run.sh <work-dir>}")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
bash "$here/racket.sh" "$work"
bash "$here/stage.sh" "$work"
bash "$here/catalog.sh" "$work"
bash "$here/install.sh" "$work"
