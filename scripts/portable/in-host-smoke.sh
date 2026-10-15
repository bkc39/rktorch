#!/usr/bin/env bash
# Runs inside a plain distro container (no Nix): install Racket from the
# racket-lang.org installer, `raco pkg install` the torch package whose
# native-libs/ carries the portable bundle, then a tensor op and 04-mlp.
# Mounts: /pkg (package source: a directory or a .zip), /installer (the
# Racket .sh installer), /scripts (this directory). PKG names the source.
set -uo pipefail

: "${PKG:?PKG names the package source under /pkg, or is catalog}"
step() { printf '\n== %s\n' "$*"; }
disk() { du -scb /root /tmp 2>/dev/null | tail -1 | cut -f1; }

step host
grep PRETTY_NAME /etc/os-release
ldd --version | head -1
ls /usr/lib64/libstdc++.so.6* /usr/lib/x86_64-linux-gnu/libstdc++.so.6* 2>/dev/null

step ca certificates
# raco reaches the catalog over https; minimal images ship no CA bundle.
if command -v apt-get >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq ca-certificates) >/dev/null 2>&1 && echo "installed ca-certificates"
fi
ls /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt 2>/dev/null

step install racket
sh /installer/racket-*.sh --in-place --dest /opt/racket >/tmp/racket-install.log 2>&1 \
  || { tail -20 /tmp/racket-install.log; exit 1; }
export PATH=/opt/racket/bin:$PATH
racket --version

step raco pkg install
before=$(disk)
( peak=0; while :; do u=$(disk); ((u > peak)) && peak=$u && echo "$peak" >/tmp/peak; sleep 1; done ) &
sampler=$!
start=$(date +%s)
if [[ $PKG == catalog ]]; then
  # A directory catalog at /pkg/catalog first, then the release's own ("").
  raco pkg config --set catalogs file:///pkg/catalog "" >/tmp/raco.log 2>&1
  raco pkg install --batch --auto ${DOCFLAG---no-docs} -j 4 torch >>/tmp/raco.log 2>&1
else
  raco pkg install --batch --auto --copy ${DOCFLAG---no-docs} -j 4 --name torch "/pkg/$PKG" >/tmp/raco.log 2>&1
fi
rc=$?
end=$(date +%s)
kill "$sampler"
grep -vE '^ *(raco setup: [0-9]+: (making|checking|compiling)|raco setup: (making|compiling))' /tmp/raco.log | tail -25
echo "raco exit $rc; wall $((end - start)) s; disk delta $(( ($(disk) - before) / 1000000 )) MB, peak $(( ($(cat /tmp/peak) - before) / 1000000 )) MB"
du -sh "$(racket -e '(display (find-system-path (quote addon-dir)))')" 2>/dev/null

step tensor smoke
racket -l racket/base -l torch -e '(define t (+ (ones 2 3) 1)) (displayln t) (displayln (item (sum t)))' 2>&1 | tail -15

step 04-mlp
racket -l racket/base -e '
(require torch/scribblings/examples/04-mlp)
(define-values (losses net device) (run-example))
(printf "device ~a\nlosses ~a\n" device losses)
(unless (< (list-ref losses 4) (list-ref losses 0)) (error "04-mlp: loss did not fall"))
(displayln "04-mlp ok")' 2>&1 | tail -15

step audio: libsndfile FLAC decode and encode
racket -l racket/base -e '
(require torch torch/audio/data)
(define fixture (collection-file-path "sine-440-16k.flac" "torch/audio/fixtures"))
(define-values (s rate) (load-audio fixture))
(save-audio "/tmp/roundtrip.flac" s rate)
(define-values (s2 rate2) (load-audio "/tmp/roundtrip.flac"))
(printf "flac ~a Hz, shape ~a -> ~a Hz, shape ~a\n" rate (shape s) rate2 (shape s2))
(unless (and (= rate rate2) (equal? (shape s) (shape s2))) (error "audio round trip differs"))
(displayln "audio ok")' 2>&1 | tail -8
