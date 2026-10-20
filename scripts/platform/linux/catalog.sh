#!/usr/bin/env bash
#
#   scripts/platform/linux/catalog.sh <platform-zip> <out-dir> [<url-root>]
set -euo pipefail

usage="usage: catalog.sh <platform-zip> <out-dir> [<url-root>]"
platform=$(realpath "${1:?$usage}")
out=$(realpath -m "${2:?$usage}")
url_root=${3:-/catalog}
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

rm -rf "$out"
mkdir -p "$out/src" "$out/zips" "$out/catalog/pkg"
git archive --prefix=torch/ HEAD:torch | tar -x -C "$out/src"
raco pkg create --format zip --dest "$out/zips" "$out/src/torch" >/dev/null
cp "$platform" "$platform.CHECKSUM" "$out/zips/"

for zip in "$out"/zips/*.zip; do
  name=$(basename "$zip" .zip)
  racket scripts/platform/catalog-entry.rkt "$name" "file://$url_root/zips/$name.zip" \
    "$zip.CHECKSUM" >"$out/catalog/pkg/$name"
done
ls -l "$out/zips"
