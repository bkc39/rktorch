#!/usr/bin/env bash
#
# Fails unless every shared library in <dir> could load on any glibc 2.28
# host: no /nix/store reference, no RPATH or RUNPATH but $ORIGIN, no GLIBC_,
# GLIBCXX_ or CXXABI_ symbol version above libtorch 2.14's own, and no NEEDED
# library outside the bundle and the system set. Prints one row per library.
#
#   scripts/platform/linux/audit-bundle.sh <dir>
set -euo pipefail

dir=${1:?usage: audit-bundle.sh <dir>}
max_glibc=2.28
max_glibcxx=3.4.22
max_cxxabi=1.3.11
system=" libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 ld-linux-x86-64.so.2 libstdc++.so.6 libgcc_s.so.1 "

failures=0
violation() {
  echo "  FAIL $1: $2"
  failures=$((failures + 1))
}

newer() {
  [[ $1 != "$2" && $(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1) == "$1" ]]
}

highest() {
  { grep -F '*UND*' <<<"$2" || true; } | { grep -oE "\\b$1[0-9.]+" || true; } |
    sed "s/^$1//" | sort -Vu | tail -1
}

mapfile -t libs < <(find "$dir" -maxdepth 1 -type f -name '*.so*' | sort)
((${#libs[@]})) || { echo "audit: no shared libraries in $dir"; exit 1; }
bundled=" $(for lib in "${libs[@]}"; do basename "$lib"; done | tr '\n' ' ') "

printf '%-28s %10s %6s %8s %7s  %s\n' library bytes GLIBC GLIBCXX CXXABI NEEDED
for lib in "${libs[@]}"; do
  name=$(basename "$lib")
  dynamic=$(readelf -d "$lib")
  symbols=$(objdump -T "$lib")
  glibc=$(highest GLIBC_ "$symbols")
  glibcxx=$(highest GLIBCXX_ "$symbols")
  cxxabi=$(highest CXXABI_ "$symbols")
  needed=$(sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' <<<"$dynamic" | tr '\n' ' ')
  printf '%-28s %10s %6s %8s %7s  %s\n' "$name" "$(stat -c %s "$lib")" \
    "${glibc:--}" "${glibcxx:--}" "${cxxabi:--}" "$needed"

  if grep -qaF /nix/store "$lib"; then
    violation "$name" "mentions $(grep -aoE '/nix/store/[^/]+' "$lib" | sort -u | head -1)"
  fi
  for entry in $(sed -n 's/.*(R\(UN\)\{0,1\}PATH).*\[\(.*\)\]/\2/p' <<<"$dynamic" | tr ':' ' '); do
    [[ $entry == '$ORIGIN' ]] || violation "$name" "runpath entry $entry"
  done
  if [[ -n $glibc ]] && newer "$glibc" "$max_glibc"; then
    violation "$name" "needs GLIBC_$glibc > $max_glibc"
  fi
  if [[ -n $glibcxx ]] && newer "$glibcxx" "$max_glibcxx"; then
    violation "$name" "needs GLIBCXX_$glibcxx > $max_glibcxx"
  fi
  if [[ -n $cxxabi ]] && newer "$cxxabi" "$max_cxxabi"; then
    violation "$name" "needs CXXABI_$cxxabi > $max_cxxabi"
  fi
  for n in $needed; do
    [[ $bundled == *" $n "* || $system == *" $n "* ]] ||
      violation "$name" "NEEDs $n, neither bundled nor a system library"
  done
done

echo "audit: NEEDED entries only; a library loaded with dlopen is neither bundled nor audited"
if ((failures)); then
  echo "audit: FAILED, $failures violations"
  exit 1
fi
echo "audit: ok (GLIBC <= $max_glibc, GLIBCXX <= $max_glibcxx, CXXABI <= $max_cxxabi)"
