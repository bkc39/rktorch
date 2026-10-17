#!/usr/bin/env bash
#
# Assembles the torch-x86_64-linux package, the skeleton under pkgs/ plus the
# libraries of a bundle build.sh made, and zips it with `raco pkg create`.
#
#   scripts/platform/linux/assemble.sh <bundle-dir> <out-dir>
#
# Writes <out-dir>/torch-x86_64-linux.zip and its .CHECKSUM.
set -euo pipefail

usage="usage: assemble.sh <bundle-dir> <out-dir>"
root=$(cd "$(dirname "$0")/../../.." && pwd)
bundle=$(realpath "${1:?$usage}")
out=$(realpath -m "${2:?$usage}")
name=torch-x86_64-linux
stage=$out/$name

rm -rf "$stage" "$out/$name.zip" "$out/$name.zip.CHECKSUM"
mkdir -p "$stage"
cp -r "$root/pkgs/$name/." "$stage/"
find "$stage" -type d -name compiled -prune -exec rm -rf {} +
cp "$bundle"/*.so* "$stage/"
raco pkg create --format zip --dest "$out" "$stage" >/dev/null

printf 'zip: %s, %.1f MB\ninstalled: %.1f MB\n' "$out/$name.zip" \
  "$(echo "$(stat -c %s "$out/$name.zip") / 1000000" | bc -l)" \
  "$(echo "$(du -sb "$stage" | cut -f1) / 1000000" | bc -l)"
