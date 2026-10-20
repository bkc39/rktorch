#!/usr/bin/env bash
#
#   [SMOKE_HOME=<dir>] [SMOKE_TESTS="<module> ..."] scripts/platform/smoke-install.sh <catalog-dir> <host-label>
set -euo pipefail

usage="usage: smoke-install.sh <catalog-dir> <host-label>"
catalog=${1:?$usage}
host=${2:?$usage}
tmp=$(mktemp -d)
step() { printf '\n== %s\n' "$*"; }

unset PLTADDONDIR TORCHRKT_NATIVE_LIB_PATH
export PLTUSERHOME=${SMOKE_HOME:-$(mktemp -d)}
mkdir -p "$PLTUSERHOME"

step "racket"
command -v racket
racket --version
echo "PLTUSERHOME=$PLTUSERHOME"

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

read -ra tests <<<"${SMOKE_TESTS:-torch/pkg-build/smoke}"
step "raco test --drdr -l ${tests[*]}"
raco test --drdr -l "${tests[@]}"

step "raco setup --check-pkg-deps --pkgs torch"
raco setup --no-docs --check-pkg-deps --pkgs torch

step "SMOKE OK: $host, racket $(racket -e '(display (version))')"
