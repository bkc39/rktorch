#!/usr/bin/env bash
#
#   scripts/platform/linux/smoke.sh <image> <racket-installer.sh> <catalog-dir>
set -euo pipefail

usage="usage: smoke.sh <image> <racket-installer.sh> <catalog-dir>"
image=${1:?$usage}
installer=$(realpath "${2:?$usage}")
catalog=$(realpath "${3:?$usage}")
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

docker run --rm --name "torchrkt-smoke-$$" --cpus 4 --memory 12g \
  -v "$catalog:/catalog:ro" -v "$installer:/installer/racket.sh:ro" \
  -v "$PWD/scripts/platform:/scripts:ro" \
  "$image" bash /scripts/linux/in-host-smoke.sh
