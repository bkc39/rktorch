#!/usr/bin/env bash
# Assemble a portable native-libs/ directory: the shim plus the NEEDED
# closure of it inside the official libtorch lib/ (so libtorch's test
# libraries, libshm and libtorch_global_deps stay out), then audit it.
#   scripts/portable/assemble-bundle.sh <libtorchrkt.so> <libtorch/lib> <out-dir> [extra.so ...]
# Extra libraries (a bundled libsndfile and its codecs) are copied as given.
# STRIP=1 drops libtorch's .symtab (--strip-unneeded): libtorch_cpu.so goes
# from 438 MB to 349 MB; the dynamic symbol table is untouched.
set -euo pipefail

shim=$1 tlib=$2 out=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd)
rm -rf "$out" && mkdir -p "$out"
cp "$shim" "$out/libtorchrkt.so"
for x in "$@"; do cp -L "$x" "$out/"; done

needed_of() { readelf -d "$1" | grep -oP 'NEEDED\).*\[\K[^]]+'; }

queue=("$out"/*.so*)
while ((${#queue[@]})); do
  lib=${queue[0]}
  queue=("${queue[@]:1}")
  for n in $(needed_of "$lib"); do
    if [[ -f $tlib/$n && ! -f $out/$n ]]; then
      if [[ ${STRIP:-0} == 1 ]]; then
        strip --strip-unneeded -o "$out/$n" "$tlib/$n"
      else
        cp "$tlib/$n" "$out/$n"
      fi
      queue+=("$out/$n")
    fi
  done
done
chmod 0644 "$out"/*

"$here/audit-bundle.sh" "$out"
du -sb "$out" | awk '{printf "uncompressed: %d bytes (%.1f MB)\n", $1, $1/1e6}'
