#!/usr/bin/env bash
#
#   scripts/platform/assemble.sh <package> <bundle-dir> <out-dir>
set -euo pipefail

usage="usage: assemble.sh <package> <bundle-dir> <out-dir>"
name=${1:?$usage}
bundle=$(realpath "${2:?$usage}")
mkdir -p "${3:?$usage}"
out=$(realpath "$3")
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
[ -f "pkgs/$name/info.rkt" ] || { echo "assemble: no skeleton pkgs/$name" >&2; exit 2; }
stage=$out/$name

rm -rf "$stage" "$out/$name.zip" "$out/$name.zip.CHECKSUM"
mkdir -p "$stage/licenses"
cp -R "pkgs/$name/." "$stage/"
cp -R scripts/platform/licenses/. "$stage/licenses/"
cp torch/LICENSE "$stage/LICENSE"
find "$stage" -type d -name compiled -prune -exec rm -rf {} +
cp "$bundle"/* "$stage/"
[ -d "$stage/licenses/pytorch-third-party" ] ||
  { echo "assemble: no licenses/pytorch-third-party in $stage" >&2; exit 2; }
cmp -s torch/LICENSE "$stage/LICENSE" ||
  { echo "assemble: $stage/LICENSE is not torch/LICENSE" >&2; exit 2; }
raco pkg create --format zip --dest "$out" "$stage" >/dev/null

megabytes() {
  find "$1" -type f -exec wc -c {} + | awk '$2 != "total" { s += $1 } END { printf "%.1f", s / 1e6 }'
}
echo "zip: $out/$name.zip, $(megabytes "$out/$name.zip") MB"
echo "installed: $(megabytes "$stage") MB"
