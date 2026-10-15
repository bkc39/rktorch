#!/usr/bin/env bash
# Load-test a portable torch package on a distro image without Nix.
#   scripts/portable/test-host.sh <image> <pkg-parent-dir> <pkg-name> <racket-installer.sh>
# <pkg-parent-dir>/<pkg-name> is a package directory or zip. The container
# is removed afterwards; the image is kept.
set -euo pipefail

image=$1 parent=$(realpath "$2") pkg=$3 installer=$(realpath "$4")
here=$(cd "$(dirname "$0")" && pwd)

docker run --rm --name "torchrkt-host-$$" --cpus 4 --memory 12g \
  -e PKG="$pkg" \
  -v "$parent:/pkg:ro" -v "$installer:/installer/$(basename "$installer"):ro" \
  -v "$here:/scripts:ro" \
  "$image" bash /scripts/in-host-smoke.sh
