#!/usr/bin/env bash
# Build libtorchrkt at the glibc 2.28 baseline in the manylinux_2_28 image.
#   scripts/portable/build-shim-manylinux.sh <libtorch-dir> <work-dir>
# <libtorch-dir> is the unpacked official libtorch 2.14.0 CPU zip (its
# top-level libtorch/ directory); <work-dir> receives the codec builds,
# logs and out/libtorchrkt.so.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
libtorch=$(realpath "$1")
work=$(realpath -m "$2")
image=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}
mkdir -p "$work"

docker run --rm --name "torchrkt-manylinux-$$" \
  --cpus 4 --memory 16g \
  -u "$(id -u):$(id -g)" -e HOME=/work -e JOBS=4 \
  -v "$repo:/src:ro" -v "$libtorch:/libtorch:ro" -v "$work:/work" \
  "$image" bash /src/scripts/portable/in-container-build.sh
