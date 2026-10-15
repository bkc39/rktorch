#!/usr/bin/env bash
# Audit a portable Linux native bundle: every shared library in <dir> must
#   - not mention /nix/store anywhere in its bytes,
#   - carry no RPATH/RUNPATH other than $ORIGIN,
#   - need no GLIBC_ symbol version newer than $MAX_GLIBC (default 2.28),
#   - need no GLIBCXX_/CXXABI_ version newer than $MAX_GLIBCXX/$MAX_CXXABI,
#   - NEED only libraries in the bundle or in the allowed system set.
#   scripts/portable/audit-bundle.sh <dir>
# Exits non-zero on any violation; prints one row per library.
set -uo pipefail

dir=${1:?usage: audit-bundle.sh <dir>}
MAX_GLIBC=${MAX_GLIBC:-2.28}
MAX_GLIBCXX=${MAX_GLIBCXX:-3.4.22}
MAX_CXXABI=${MAX_CXXABI:-1.3.11}
ALLOWED_SYSTEM=${ALLOWED_SYSTEM:-"libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 ld-linux-x86-64.so.2 libstdc++.so.6 libgcc_s.so.1"}

fail=0
violation() { echo "  FAIL $1: $2"; fail=1; }

newer() { # newer <have> <max>: true when have > max
  [[ $1 != "$2" && $(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1) == "$1" ]]
}

max_version() { # max_version <prefix> <objdump -T output>
  grep -E '\*UND\*' <<<"$2" | grep -oE "\(?$1[0-9.]+" | sed -E "s/^\(?$1//" | sort -Vu | tail -1
}

mapfile -t libs < <(find "$dir" -maxdepth 1 -type f -name '*.so*' | sort)
[[ ${#libs[@]} -gt 0 ]] || { echo "no shared libraries in $dir"; exit 1; }
bundled=" $(for l in "${libs[@]}"; do basename "$l"; done | tr '\n' ' ') "

printf '%-40s %10s %7s %9s %8s  %s\n' library bytes GLIBC GLIBCXX CXXABI NEEDED-outside-bundle
for lib in "${libs[@]}"; do
  name=$(basename "$lib")
  dyn=$(readelf -d "$lib" 2>&1)
  syms=$(objdump -T "$lib" 2>&1)
  glibc=$(max_version GLIBC_ "$syms")
  glibcxx=$(max_version GLIBCXX_ "$syms")
  cxxabi=$(max_version CXXABI_ "$syms")
  needed=$(grep -oP 'NEEDED\).*\[\K[^]]+' <<<"$dyn")
  outside=$(for n in $needed; do [[ $bundled == *" $n "* ]] || echo "$n"; done | tr '\n' ' ')
  printf '%-40s %10s %7s %9s %8s  %s\n' "$name" "$(stat -c %s "$lib")" \
    "${glibc:--}" "${glibcxx:--}" "${cxxabi:--}" "$outside"

  grep -qa /nix/store "$lib" && violation "$name" "mentions /nix/store: $(grep -ao '/nix/store/[^/]*' "$lib" | sort -u | head -3 | tr '\n' ' ')"
  for rp in $(grep -oP '(RPATH|RUNPATH)\).*\[\K[^]]+' <<<"$dyn" | tr ':' ' '); do
    [[ $rp == '$ORIGIN' ]] || violation "$name" "runpath entry $rp"
  done
  [[ -n $glibc ]] && newer "$glibc" "$MAX_GLIBC" && violation "$name" "needs GLIBC_$glibc > $MAX_GLIBC"
  [[ -n $glibcxx ]] && newer "$glibcxx" "$MAX_GLIBCXX" && violation "$name" "needs GLIBCXX_$glibcxx > $MAX_GLIBCXX"
  [[ -n $cxxabi ]] && newer "$cxxabi" "$MAX_CXXABI" && violation "$name" "needs CXXABI_$cxxabi > $MAX_CXXABI"
  for n in $outside; do
    [[ " $ALLOWED_SYSTEM " == *" $n "* ]] || violation "$name" "NEEDs $n, neither bundled nor allowed"
  done
done

if [[ $fail -ne 0 ]]; then
  echo "audit: FAILED"
  exit 1
fi
echo "audit: ok (glibc <= $MAX_GLIBC, GLIBCXX <= $MAX_GLIBCXX, CXXABI <= $MAX_CXXABI)"
