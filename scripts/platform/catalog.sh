#!/usr/bin/env bash
#
#   scripts/platform/catalog.sh <platform-zip> <out-dir> [<url-root>] [<platform-https-root>]
set -euo pipefail

usage="usage: catalog.sh <platform-zip> <out-dir> [<url-root>] [<platform-https-root>]"
platform=$(realpath "${1:?$usage}")
mkdir -p "${2:?$usage}"
out=$(realpath "$2")
url_root=${3:-/catalog}
platform_root=${4:-}
platform_name=$(basename "$platform" .zip)
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

rm -rf "$out"
mkdir -p "$out/src" "$out/zips" "$out/catalog/pkg"
git archive --prefix=torch/ HEAD:torch | tar -x -C "$out/src"
raco pkg create --format zip --dest "$out/zips" "$out/src/torch" >/dev/null
cp "$platform" "$platform.CHECKSUM" "$out/zips/"

for zip in "$out"/zips/*.zip; do
  name=$(basename "$zip" .zip)
  src=file://$url_root/zips/$name.zip
  if [ -n "$platform_root" ] && [ "$name" = "$platform_name" ]; then
    src=$platform_root/$name.zip
  fi
  racket scripts/platform/catalog-entry.rkt "$name" "$src" "$zip" >"$out/catalog/pkg/$name"
done
ls -l "$out/zips"
