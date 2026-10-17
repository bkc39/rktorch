#!/usr/bin/env bash
#
#   bash /scripts/linux/in-host-smoke.sh   (run by smoke.sh)
set -euo pipefail

distro=$(. /etc/os-release && echo "$ID-$VERSION_ID")
step() { printf '\n== %s\n' "$*"; }

step "$distro, $(ldd --version | head -1)"
if command -v apt-get >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq ca-certificates libsqlite3-0) >/dev/null 2>&1
fi

step "install Racket"
sh /installer/racket.sh --in-place --dest /opt/racket >/tmp/racket-install.log 2>&1 ||
  { tail -20 /tmp/racket-install.log; exit 1; }
export PATH=/opt/racket/bin:$PATH
racket --version

step "raco pkg install --no-docs torch"
raco pkg config --set catalogs file:///catalog/catalog ""
start=$(date +%s)
raco pkg install --batch --auto --no-docs -j 4 torch >/tmp/raco.out 2>/tmp/raco.err
echo "wall $(($(date +%s) - start)) s"
grep -E '^Inferred|^Downloading|^raco setup: (pre-installing|--- compiling)' /tmp/raco.out || true
du -sh "$(racket -e '(display (find-system-path (quote addon-dir)))')"
if [ -s /tmp/raco.err ]; then
  echo "stderr was not empty:"
  cat /tmp/raco.err
  exit 1
fi
echo "stderr: empty"

step "04-mlp"
raco test /scripts/mlp-smoke.rkt

step "FLAC round trip"
raco test /scripts/flac-round-trip.rkt

step "SMOKE OK: $distro, racket $(racket -e '(display (version))')"
