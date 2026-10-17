#!/usr/bin/env bash
#
# The container half of smoke.sh, which mounts the catalog at /catalog, the
# Racket installer at /installer/racket.sh and this directory at /scripts.
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
if [[ -s /tmp/raco.err ]]; then
  echo "stderr was not empty:"
  cat /tmp/raco.err
  exit 1
fi
echo "stderr: empty"

step "04-mlp"
racket -l racket/base -e '
(require torch torch/scribblings/examples/04-mlp
         (only-in torch/foreign/raw/native-location native-library-directory))
(printf "shim from ~a\n" (native-library-directory))
(define-values (losses net device) (run-example))
(printf "device ~a, losses ~a\n" device losses)
(unless (< (list-ref losses 4) (list-ref losses 0)) (error "04-mlp: the loss did not fall"))'

step "FLAC round trip"
racket -l racket/base -e '
(require torch torch/audio/data)
(define fixture (collection-file-path "sine-440-16k.flac" "torch/audio/fixtures"))
(define-values (s rate) (load-audio fixture))
(save-audio "/tmp/roundtrip.flac" s rate)
(define-values (s2 rate2) (load-audio "/tmp/roundtrip.flac"))
(printf "~a Hz ~a -> ~a Hz ~a\n" rate (shape s) rate2 (shape s2))
(unless (and (= rate rate2) (equal? (shape s) (shape s2))) (error "the round trip differs"))'

step "SMOKE OK: $distro, racket $(racket -e '(display (version))')"
