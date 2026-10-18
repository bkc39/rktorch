#!/usr/bin/env bash
#
#   scripts/platform/smoke-install.sh <catalog-dir> <host-label>
set -euo pipefail

usage="usage: smoke-install.sh <catalog-dir> <host-label>"
catalog=${1:?$usage}
host=${2:?$usage}
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
step() { printf '\n== %s\n' "$*"; }

step "racket"
command -v racket
racket --version

step "raco pkg install --no-docs torch"
raco pkg config --set catalogs "file://$catalog/catalog" ""
start=$(date +%s)
if ! raco pkg install --batch --auto --no-docs -j 4 torch >"$tmp/raco.out" 2>"$tmp/raco.err"; then
  tail -40 "$tmp/raco.out"
  cat "$tmp/raco.err"
  echo "raco pkg install failed"
  exit 1
fi
echo "wall $(($(date +%s) - start)) s"
grep -E '^Inferred|^Downloading|^raco setup: (pre-installing|--- compiling)' "$tmp/raco.out" || true
du -sh "$(racket -e '(display (find-system-path (quote addon-dir)))')"
if [ -s "$tmp/raco.err" ]; then
  echo "stderr was not empty:"
  cat "$tmp/raco.err"
  exit 1
fi
echo "stderr: empty"

step "04-mlp"
raco test "$here/mlp-smoke.rkt"

step "FLAC round trip"
raco test "$here/flac-round-trip.rkt"

step "SMOKE OK: $host, racket $(racket -e '(display (version))')"
