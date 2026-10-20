#!/usr/bin/env bash
#
#   scripts/platform/macos/audit-test.sh <bundle-dir> <libtorch-lib-dir>
set -euo pipefail

usage="usage: audit-test.sh <bundle-dir> <libtorch-lib-dir>"
bundle=${1:?$usage}
zip_libs=${2:?$usage}
here=$(cd "$(dirname "$0")" && pwd)
target=$(cat "$here/deployment-target")
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

audit_fails() {
  if "$here/audit-bundle.sh" "$@" >"$scratch/log" 2>&1; then
    echo "audit-test: the audit passed $*" >&2
    cat "$scratch/log" >&2
    exit 1
  fi
}

expect() {
  grep -qF -- "$1" "$scratch/log" || {
    echo "audit-test: no \"$1\" in" >&2
    cat "$scratch/log" >&2
    exit 1
  }
  echo "ok: $1"
}

audit_fails "$zip_libs" "$target"
expect "FAIL libomp.dylib: id /opt/llvm-openmp/lib/libomp.dylib, not @rpath/libomp.dylib"
audit_fails "$zip_libs" 13.0
expect "FAIL libc10.dylib: minos 14.0 > 13.0"

mkdir "$scratch/tampered"
shim=$scratch/tampered/libtorchrkt.dylib
cp "$bundle/libtorchrkt.dylib" "$shim"
loads=()
while read -r path; do
  loads+=("$path")
done < <("$here/load-commands.sh" "$shim" | awk '$1 != "LC_ID_DYLIB" && $2 ~ /^@rpath\// { print $2 }')
[ "${#loads[@]}" -ge 3 ] || { echo "audit-test: the shim loads fewer than 3 @rpath libraries: ${loads[*]}" >&2; exit 1; }
install_name_tool -id /usr/local/lib/libtorchrkt.dylib \
  -change "${loads[0]}" "/opt/homebrew/lib/${loads[0]#@rpath/}" \
  -change "${loads[1]}" /usr/lib/libcurl.4.dylib \
  -add_rpath /nix/store/00000000000000000000000000000000-audit-test/lib \
  "$shim"
codesign --remove-signature "$shim"
audit_fails "$scratch/tampered" "$target"
expect "FAIL libtorchrkt.dylib: mentions /nix/store/00000000000000000000000000000000-audit-test"
expect "FAIL libtorchrkt.dylib: id /usr/local/lib/libtorchrkt.dylib, not @rpath/libtorchrkt.dylib"
expect "FAIL libtorchrkt.dylib: LC_RPATH /nix/store/00000000000000000000000000000000-audit-test/lib"
expect "FAIL libtorchrkt.dylib: 2 LC_RPATH entries"
expect "FAIL libtorchrkt.dylib: loads /opt/homebrew/lib/${loads[0]#@rpath/}, outside @rpath, @loader_path, /usr/lib and /System"
expect "FAIL libtorchrkt.dylib: loads /usr/lib/libcurl.4.dylib, neither bundled nor allowed"
expect "FAIL libtorchrkt.dylib: loads ${loads[2]}, which is not in the bundle"
expect "FAIL libtorchrkt.dylib: codesign --verify:"

rm "$shim"
cp "$bundle/libtorchrkt.dylib" "$shim"
install_name_tool -delete_rpath @loader_path "$shim"
codesign --force --sign - "$shim"
cp "$bundle/${loads[0]#@rpath/}" "$scratch/tampered/"
audit_fails "$scratch/tampered" "$target"
expect "FAIL libtorchrkt.dylib: loads bundled libraries through @rpath, but has no LC_RPATH @loader_path"
echo "audit-test: ok"
