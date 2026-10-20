#!/usr/bin/env bash
#
#   scripts/platform/macos/audit-bundle.sh <dir> <deployment-target>
set -euo pipefail

usage="usage: audit-bundle.sh <dir> <deployment-target>"
dir=${1:?$usage}
target=${2:?$usage}
here=$(cd "$(dirname "$0")" && pwd)
system=" /usr/lib/libSystem.B.dylib /usr/lib/libc++.1.dylib /usr/lib/libobjc.A.dylib "

failures=0
violation() {
  echo "  FAIL $1: $2"
  failures=$((failures + 1))
}

above() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    n = split(a, x, "."); m = split(b, y, ".")
    for (i = 1; i <= (n > m ? n : m); i++)
      if (x[i] + 0 != y[i] + 0) exit !(x[i] + 0 > y[i] + 0)
    exit 1 }'
}

libs=()
while IFS= read -r lib; do
  libs+=("$lib")
done < <(find "$dir" -maxdepth 1 -type f -name '*.dylib' | sort)
[ "${#libs[@]}" -gt 0 ] || { echo "audit: no dylibs in $dir"; exit 1; }

printf '%-28s %10s %6s  %s\n' library bytes minos 'loads (outside /usr/lib and /System)'
for lib in "${libs[@]}"; do
  name=$(basename "$lib")
  commands=$("$here/load-commands.sh" "$lib")
  id=$(awk '$1 == "LC_ID_DYLIB" { print $2 }' <<<"$commands")
  minos=$(awk '$1 == "LC_BUILD_VERSION" || $1 == "LC_VERSION_MIN_MACOSX" { print $2; exit }' <<<"$commands")
  rpaths=$(awk '$1 == "LC_RPATH" { print $2 }' <<<"$commands")
  loads=$(awk '$1 ~ /_DYLIB$/ && $1 != "LC_ID_DYLIB" { print $2 }' <<<"$commands")
  printf '%-28s %10s %6s  %s\n' "$name" "$(wc -c <"$lib" | tr -d ' ')" "${minos:--}" \
    "$(grep -vE '^(/usr/lib|/System)/' <<<"$loads" | tr '\n' ' ' || true)"

  if grep -qaF /nix/store "$lib"; then
    violation "$name" "mentions $(grep -aoE '/nix/store/[^/]+' "$lib" | sort -u | head -1)"
  fi
  [ "$id" = "@rpath/$name" ] || violation "$name" "id ${id:-<none>}, not @rpath/$name"
  if [ -z "$minos" ]; then
    violation "$name" "no LC_BUILD_VERSION minos"
  elif above "$minos" "$target"; then
    violation "$name" "minos $minos > $target"
  fi
  count=0
  for rpath in $rpaths; do
    count=$((count + 1))
    case $rpath in
      @loader_path | @loader_path/) ;;
      *) violation "$name" "LC_RPATH $rpath" ;;
    esac
  done
  [ "$count" -le 1 ] || violation "$name" "$count LC_RPATH entries"
  via_rpath=no
  for path in $loads; do
    case $path in
      @rpath/* | @loader_path/*)
        if [ -f "$dir/${path#*/}" ]; then
          case $path in @rpath/*) via_rpath=yes ;; esac
        else
          violation "$name" "loads $path, which is not in the bundle"
        fi
        ;;
      /System/*) ;;
      /usr/lib/*)
        case $system in
          *" $path "*) ;;
          *) violation "$name" "loads $path, neither bundled nor allowed" ;;
        esac
        ;;
      *) violation "$name" "loads $path, outside @rpath, @loader_path, /usr/lib and /System" ;;
    esac
  done
  if [ "$via_rpath" = yes ] && [ "$count" -eq 0 ]; then
    violation "$name" "loads bundled libraries through @rpath, but has no LC_RPATH @loader_path"
  fi
  if ! signature=$(codesign --verify "$lib" 2>&1); then
    violation "$name" "codesign --verify: $(head -1 <<<"$signature")"
  fi
done

if [ "$failures" -gt 0 ]; then
  echo "audit: FAILED, $failures violations"
  exit 1
fi
echo "audit: ok (minos <= $target, ids @rpath, LC_RPATH @loader_path, signatures valid)"
