#!/usr/bin/env bash
#
#   scripts/platform/linux/assemble.sh <bundle-dir> <out-dir>
set -euo pipefail

usage="usage: assemble.sh <bundle-dir> <out-dir>"
bundle=$(realpath "${1:?$usage}")
out=$(realpath -m "${2:?$usage}")
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
name=torch-x86_64-linux
stage=$out/$name

rm -rf "$stage" "$out/$name.zip" "$out/$name.zip.CHECKSUM"
mkdir -p "$stage"
cp -r "pkgs/$name/." "$stage/"
find "$stage" -type d -name compiled -prune -exec rm -rf {} +
cp "$bundle"/*.so* "$stage/"
raco pkg create --format zip --dest "$out" "$stage" >/dev/null

{ stat -c %s "$out/$name.zip"; du -sb "$stage" | cut -f1; } |
  awk -v zip="$out/$name.zip" '
    NR == 1 { printf "zip: %s, %.1f MB\n", zip, $1 / 1e6 }
    NR == 2 { printf "installed: %.1f MB\n", $1 / 1e6 }'
