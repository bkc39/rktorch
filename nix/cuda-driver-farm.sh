#!/usr/bin/env bash
# Links the host NVIDIA driver libraries into <dir>: nix's glibc never reads
# the host's ld.so.cache. Exits 1 when any is missing.
#   cuda-driver-farm.sh <dir>
set -uo pipefail

farm=$1
rm -rf "$farm"
mkdir -p "$farm"
missing=0
for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
  path=$(/sbin/ldconfig -p 2>/dev/null | grep -F "$lib" | grep -oE '/[^ ]+' | head -1)
  if [ -n "$path" ]; then
    ln -sf "$path" "$farm/$lib"
  else
    echo "WARNING: $lib not found via ldconfig; CUDA calls may fail" >&2
    missing=1
  fi
done
exit "$missing"
