#!/usr/bin/env bash
#
#   scripts/platform/linux/build.sh <work-dir>
set -euo pipefail

work=$(realpath -m "${1:?usage: build.sh <work-dir>}")
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
image=quay.io/pypa/manylinux_2_28_x86_64@sha256:4131ddd7acb9c6c966af02fe7e71edc51e078c2d41f475966697676430bfa2ae
mkdir -p "$work"

docker run --rm --name "torchrkt-manylinux-$$" \
  --cpus 4 --memory 16g \
  -u "$(id -u):$(id -g)" -e HOME=/work -e JOBS=4 -e IMAGE="$image" \
  -v "$PWD:/src:ro" -v "$work:/work" \
  "$image" bash /src/scripts/platform/linux/in-container.sh
