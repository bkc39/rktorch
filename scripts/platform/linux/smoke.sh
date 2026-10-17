#!/usr/bin/env bash
#
# Installs torch on a distro image without Nix, from a catalog catalog.sh
# made and a Racket installer from download.racket-lang.org, then trains
# 04-mlp and round-trips a FLAC file. Fails if `raco pkg install` writes
# anything to stderr: the package server counts that as a failure.
#
#   scripts/platform/linux/smoke.sh <image> <racket-installer.sh> <catalog-dir>
set -euo pipefail

usage="usage: smoke.sh <image> <racket-installer.sh> <catalog-dir>"
image=${1:?$usage}
installer=$(realpath "${2:?$usage}")
catalog=$(realpath "${3:?$usage}")
here=$(cd "$(dirname "$0")" && pwd)

docker run --rm --name "torchrkt-smoke-$$" --cpus 4 --memory 12g \
  -v "$catalog:/catalog:ro" -v "$installer:/installer/racket.sh:ro" \
  -v "$here:/scripts:ro" \
  "$image" bash /scripts/in-host-smoke.sh
