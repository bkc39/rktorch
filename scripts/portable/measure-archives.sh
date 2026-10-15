#!/usr/bin/env bash
# Compressed sizes of a bundle directory: zip -9, tar.gz -9, tar.xz -9, and
# the zip `raco pkg create` writes for a package directory holding it.
#   scripts/portable/measure-archives.sh <bundle-dir> <out-dir> [<pkg-dir>]
set -euo pipefail

bundle=$(realpath "$1") out=$(realpath -m "$2") pkg=${3:-}
mkdir -p "$out"
rm -f "$out"/bundle.*
row() { printf '%-28s %12d bytes  %7.1f MB  %5.1f s\n' "$1" "$(stat -c %s "$2")" "$(echo "$(stat -c %s "$2")/1000000" | bc -l)" "$3"; }
timed() { local s=$(date +%s.%N); "$@" >/dev/null; echo "$(date +%s.%N) - $s" | bc -l; }

t=$(cd "$(dirname "$bundle")" && timed zip -9 -r -q "$out/bundle.zip" "$(basename "$bundle")")
row zip-9 "$out/bundle.zip" "$t"
t=$(timed tar -C "$(dirname "$bundle")" -czf "$out/bundle.tgz" "$(basename "$bundle")")
row tar.gz "$out/bundle.tgz" "$t"
t=$(timed tar -C "$(dirname "$bundle")" -cJf "$out/bundle.txz" "$(basename "$bundle")")
row tar.xz-6 "$out/bundle.txz" "$t"
if [[ -n $pkg ]]; then
  rm -rf "$out/raco" && mkdir -p "$out/raco"
  t=$(timed raco pkg create --format zip --dest "$out/raco" "$pkg")
  row "raco pkg create (zip)" "$out/raco/$(basename "$pkg").zip" "$t"
fi
