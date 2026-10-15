#!/usr/bin/env bash
# nix's glibc never reads the host's ld.so.cache.
set -uo pipefail

farm=$1
IFS=: read -ra dirs <<<"${TORCHRKT_DRIVER_DIRS:-/run/opengl-driver/lib}"
if [ -n "${LDCONFIG:-}" ]; then
  ldconfigs=("$LDCONFIG")
else
  ldconfigs=(ldconfig /sbin/ldconfig)
fi

lookup() {
  local dir ldconfig path
  for dir in "${dirs[@]}"; do
    [ -e "$dir/$1" ] && { echo "$dir/$1"; return; }
  done
  for ldconfig in "${ldconfigs[@]}"; do
    path=$("$ldconfig" -p 2>/dev/null | grep -F "$1" | grep -oE '/[^ ]+' | head -1)
    [ -z "$path" ] || { echo "$path"; return; }
  done
}

rm -rf "$farm"
mkdir -p "$farm"
missing=0
for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
  path=$(lookup "$lib")
  if [ -n "$path" ]; then
    ln -sf "$path" "$farm/$lib"
  else
    echo "WARNING: $lib not found in ${dirs[*]} or via ldconfig; CUDA calls may fail" >&2
    missing=1
  fi
done
exit "$missing"
