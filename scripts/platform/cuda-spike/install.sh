#!/usr/bin/env bash
#
#   scripts/platform/cuda-spike/install.sh <work-dir>
#
# Installs torch from <work-dir>/catalog-dir into the stock Racket in
# <work-dir>/racket, in a cleared environment: no Nix on PATH, no
# LD_LIBRARY_PATH, a fresh PLTUSERHOME. Then trains on 'cuda.
set -euo pipefail

work=$(realpath "${1:?usage: install.sh <work-dir>}")
here=$(cd "$(dirname "$0")" && pwd)
home=$work/home
rm -rf "$home"
mkdir -p "$home"
clean=(env -i HOME="$home" PATH="$work/racket/bin:/usr/bin:/bin" LANG=C.UTF-8 SMOKE_HOME="$home")

"${clean[@]}" bash "$here/../smoke-install.sh" "$work/catalog-dir" "lab host, stock Racket 9.1, no Nix"

printf '\n== the CUDA smoke, LD_DEBUG=libs to %s\n' "$work/ld-debug.log"
start=$(date +%s)
"${clean[@]}" LD_DEBUG=libs LD_DEBUG_OUTPUT="$work/ld-debug" racket "$here/cuda-smoke.rkt"
echo "wall $(($(date +%s) - start)) s"
cat "$work"/ld-debug.* >"$work/ld-debug.log"
rm -f "$work"/ld-debug.[0-9]*
grep -oE 'calling init: [^ ]+' "$work/ld-debug.log" | sort -u | sed 's/calling init: /  loaded /'
