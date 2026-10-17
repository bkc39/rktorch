#!/usr/bin/env bash
#
# A directory catalog for the no-Nix smoke: torch, zipped from this
# checkout's HEAD, and a platform package zip assemble.sh made. Its sources
# are file:// URLs under <url-root>, where the smoke mounts <out-dir>.
#
#   scripts/platform/linux/catalog.sh <platform-zip> <out-dir> [<url-root>]
set -euo pipefail

usage="usage: catalog.sh <platform-zip> <out-dir> [<url-root>]"
root=$(cd "$(dirname "$0")/../../.." && pwd)
platform=$(realpath "${1:?$usage}")
out=$(realpath -m "${2:?$usage}")
url_root=${3:-/catalog}

rm -rf "$out"
mkdir -p "$out/src" "$out/zips" "$out/catalog/pkg"
git -C "$root" archive --prefix=torch/ HEAD:torch | tar -x -C "$out/src"
raco pkg create --format zip --dest "$out/zips" "$out/src/torch" >/dev/null
cp "$platform" "$platform.CHECKSUM" "$out/zips/"

for zip in "$out"/zips/*.zip; do
  name=$(basename "$zip" .zip)
  printf '#hash((name . "%s") (source . "file://%s/zips/%s.zip") (checksum . "%s") (author . "") (description . "") (tags . ()) (dependencies . ()) (modules . ()))\n' \
    "$name" "$url_root" "$name" "$(cat "$zip.CHECKSUM")" >"$out/catalog/pkg/$name"
done
ls -l "$out/zips"
