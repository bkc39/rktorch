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

bash /scripts/smoke-install.sh /catalog "$distro"
