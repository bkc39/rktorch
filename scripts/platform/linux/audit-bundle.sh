#!/usr/bin/env bash
#
#   scripts/platform/linux/audit-bundle.sh <dir>
set -euo pipefail

dir=${1:?usage: audit-bundle.sh <dir>}
limits='GLIBC 2.28
GLIBCXX 3.4.22
CXXABI 1.3.11'
system=" libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 ld-linux-x86-64.so.2 libstdc++.so.6 libgcc_s.so.1 "

failures=0
violation() {
  echo "  FAIL $1: $2"
  failures=$((failures + 1))
}

newer() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]
}

highest() {
  { grep -oE "\\b$1_[0-9.]+" <<<"$2" || true; } | sed "s/^$1_//" | sort -Vu | tail -1
}

mapfile -t libs < <(find "$dir" -maxdepth 1 -type f -name '*.so*' | sort)
[ "${#libs[@]}" -gt 0 ] || { echo "audit: no shared libraries in $dir"; exit 1; }
bundled=" $(for lib in "${libs[@]}"; do basename "$lib"; done | tr '\n' ' ') "

printf '%-28s %10s' library bytes
while read -r prefix _; do printf ' %8s' "$prefix"; done <<<"$limits"
printf '  %s\n' NEEDED
for lib in "${libs[@]}"; do
  name=$(basename "$lib")
  dynamic=$(readelf -d "$lib")
  undefined=$(objdump -T "$lib" | grep -F '*UND*' || true)
  needs=$(readelf -V "$lib" | sed -n '/^Version needs section/,/^$/p')
  needed=$(sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' <<<"$dynamic" | tr '\n' ' ')
  runpath=$(sed -n 's/.*(R\(UN\)\{0,1\}PATH).*\[\(.*\)\]/\2/p' <<<"$dynamic" | tr ':' ' ')

  declare -A seen=() listed=()
  printf '%-28s %10s' "$name" "$(stat -c %s "$lib")"
  while read -r prefix _; do
    seen[$prefix]=$(highest "$prefix" "$undefined")
    listed[$prefix]=$(highest "$prefix" "$needs")
    printf ' %8s' "${seen[$prefix]:--}"
  done <<<"$limits"
  printf '  %s\n' "$needed"

  if grep -qaF /nix/store "$lib"; then
    violation "$name" "mentions $(grep -aoE '/nix/store/[^/]+' "$lib" | sort -u | head -1)"
  fi
  for entry in $runpath; do
    [ "$entry" = '$ORIGIN' ] || violation "$name" "runpath entry $entry"
  done
  while read -r prefix max; do
    [ "${seen[$prefix]}" = "${listed[$prefix]}" ] ||
      violation "$name" "objdump -T reads ${prefix}_${seen[$prefix]:-<none>}, readelf -V ${prefix}_${listed[$prefix]:-<none>}"
    if [ -n "${seen[$prefix]}" ] && newer "${seen[$prefix]}" "$max"; then
      violation "$name" "needs ${prefix}_${seen[$prefix]} > $max"
    fi
  done <<<"$limits"
  inside=()
  for n in $needed; do
    case $bundled in
      *" $n "*) inside+=("$n") ;;
      *)
        case $system in
          *" $n "*) ;;
          *) violation "$name" "NEEDs $n, neither bundled nor a system library" ;;
        esac
        ;;
    esac
  done
  case " $runpath " in
    *' $ORIGIN '*) ;;
    *) [ "${#inside[@]}" -eq 0 ] ||
         violation "$name" "NEEDs bundled ${inside[*]}, but its runpath lacks \$ORIGIN" ;;
  esac
done

echo "audit: NEEDED entries only; a library loaded with dlopen is neither bundled nor audited"
if [ "$failures" -gt 0 ]; then
  echo "audit: FAILED, $failures violations"
  exit 1
fi
echo "audit: ok ($(awk '{ printf "%s%s <= %s", (NR > 1 ? ", " : ""), $1, $2 }' <<<"$limits"))"
