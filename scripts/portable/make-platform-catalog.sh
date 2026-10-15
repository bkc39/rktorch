#!/usr/bin/env bash
# The platform-package layout, prototyped: torch (sources only) depends on
# torch-x86_64-linux with #:platform, and the platform package carries the
# bundle as collection "torch-native". Both are zipped with `raco pkg create`
# and listed in a directory catalog served from file://.
#   scripts/portable/make-platform-catalog.sh <torch-src-dir> <bundle-dir> <out-dir> [<catalog-url-root>]
# <catalog-url-root> is where <out-dir> will be mounted (default: out-dir).
set -euo pipefail

src=$(realpath "$1") bundle=$(realpath "$2") out=$(realpath -m "$3")
root=${4:-$out}
here=$(cd "$(dirname "$0")" && pwd)
rm -rf "$out" && mkdir -p "$out/src" "$out/zips" "$out/catalog/pkg"

plat=$out/src/torch-x86_64-linux
mkdir -p "$plat"
cp "$bundle"/*.so* "$plat/"
cat >"$plat/info.rkt" <<'EOF'
#lang info
(define collection "torch-native")
(define deps '("base"))
(define pkg-desc "libtorchrkt and libtorch 2.14.0 (CPU) for x86_64 Linux")
EOF

cp -r "$src" "$out/src/torch"
rm -rf "$out/src/torch/native-libs" "$out/src/torch/compiled"
python3 -I "$here/platform-loader-patch.py" "$out/src/torch"

for p in torch-x86_64-linux torch; do
  /usr/bin/time -f "raco pkg create $p: %e s, %M KB max RSS" \
    raco pkg create --format zip --dest "$out/zips" "$out/src/$p" >/dev/null
  sum=$(cat "$out/zips/$p.zip.CHECKSUM")
  printf '#hash((source . "file://%s/zips/%s.zip") (checksum . "%s") (name . "%s") (author . "") (description . "") (tags . ()) (dependencies . ()) (modules . ()))\n' \
    "$root" "$p" "$sum" "$p" >"$out/catalog/pkg/$p"
done
ls -l "$out/zips"
