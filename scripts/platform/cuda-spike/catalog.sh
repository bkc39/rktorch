#!/usr/bin/env bash
#
#   scripts/platform/cuda-spike/catalog.sh <work-dir>   (a directory catalog in <work-dir>/catalog-dir)
set -euo pipefail

work=$(realpath "${1:?usage: catalog.sh <work-dir>}")
here=$(cd "$(dirname "$0")" && pwd)
bin=$work/racket/bin
out=$work/catalog-dir
cd "$(git -C "$here" rev-parse --show-toplevel)"

rm -rf "$out"
mkdir -p "$out/src" "$out/zips" "$out/catalog/pkg"
git archive --prefix=torch/ HEAD:torch | tar -x -C "$out/src"
"$bin/raco" pkg create --format zip --dest "$out/zips" "$out/src/torch" >/dev/null
for zip in "$work"/zips/*.zip; do
  ln -s "$zip" "$out/zips/"
  ln -s "$zip.CHECKSUM" "$out/zips/"
done

for zip in "$out"/zips/*.zip; do
  name=$(basename "$zip" .zip)
  "$bin/racket" scripts/platform/catalog-entry.rkt "$name" "file://$out/zips/$name.zip" "$zip" \
    >"$out/catalog/pkg/$name"
done
ls -lL "$out/zips"
