#!/usr/bin/env bash
#
# Builds the x86_64 Linux native bundle in the manylinux_2_28 image: the shim
# at the glibc 2.28 baseline with RUNPATH $ORIGIN, beside libtorch and the
# shared libsndfile it loads. The audit fails the build, and the shim's
# GoogleTest suite runs against the bundle. See docs/building.md.
#
#   scripts/platform/linux/build.sh <work-dir>
#
# <work-dir> keeps the downloads and builds between runs; the bundle lands in
# <work-dir>/bundle. Downloads already in <work-dir>/dl are checked, not
# refetched.
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
work=$(realpath -m "${1:?usage: build.sh <work-dir>}")
image=quay.io/pypa/manylinux_2_28_x86_64@sha256:4131ddd7acb9c6c966af02fe7e71edc51e078c2d41f475966697676430bfa2ae
mkdir -p "$work"

docker run --rm --name "torchrkt-manylinux-$$" \
  --cpus 4 --memory 16g \
  -u "$(id -u):$(id -g)" -e HOME=/work -e JOBS=4 \
  -v "$root:/src:ro" -v "$work:/work" \
  "$image" bash /src/scripts/platform/linux/in-container.sh
