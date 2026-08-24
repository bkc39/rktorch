#!/usr/bin/env bash
# Reproduce issue #72's vector: overwrite the mapped libtorchrkt.dylib IN PLACE
# under a live REPL that holds tensors, then touch native code and send EOF
# (== Ctrl-D).  Restores the library from the nix store on the way out.
#
# Usage: scripts/repro/corrupt-restage.sh [mode]
#   mode = zero    (default) zero 64K of the __TEXT segment -- what a partial
#                  `cp` over a mapped file does
#   mode = restage re-`cp` the identical bytes in place -- what
#                  `nix run .#copy-native-libs` actually does
set -u
mode="${1:-zero}"
root=/Users/bkc/dev/rkt/rktorch
lib="$root/torch/native-libs/libtorchrkt.dylib"
out="${REPRO_LOGDIR:-${TMPDIR:-/tmp}}/corrupt-$mode.log"
backup="${TMPDIR:-/tmp}/libtorchrkt.dylib.bak"

cp "$lib" "$backup"
fifo=$(mktemp -u); mkfifo "$fifo"

racket -i < "$fifo" > "$out" 2>&1 &
rkt=$!
exec 3> "$fifo"

echo "[repro] racket pid=$rkt log=$out"
printf '(require torch)\n' >&3
printf '(define held (for/list ([_ (in-range 500)]) (randn 256 256)))\n' >&3
printf '(length held)\n' >&3
sleep 8

echo "[repro] corrupting mapped dylib in place (mode=$mode)"
case "$mode" in
  zero)    dd if=/dev/zero of="$lib" bs=4096 seek=4 count=16 conv=notrunc 2>/dev/null ;;
  restage) cp "$backup" "$lib" ;;
esac

sleep 2
echo "[repro] touching native code post-corruption"
printf '(randn 8 8)\n' >&3
printf '(car held)\n' >&3
sleep 5

echo "[repro] sending EOF (Ctrl-D)"
exec 3>&-
for _ in $(seq 1 40); do kill -0 "$rkt" 2>/dev/null || break; sleep 1; done

if kill -0 "$rkt" 2>/dev/null; then
  echo "[repro] STILL ALIVE after 40s -- capturing state"
  sample "$rkt" 3 -f "${out%.log}-sample.txt" 2>/dev/null || true
  ps -o rss=,vsz= -p "$rkt"
  echo "[repro] SIGTERM"; kill -TERM "$rkt" 2>/dev/null; sleep 5
  if kill -0 "$rkt" 2>/dev/null; then echo "[repro] TERM IGNORED -> SIGKILL"; kill -9 "$rkt"; fi
fi
wait "$rkt" 2>/dev/null; code=$?
rm -f "$fifo"
cp "$backup" "$lib"; chmod u+w "$lib"

echo "[repro] exit=$code  logbytes=$(wc -c < "$out")"
grep -c 'invalid memory reference' "$out" 2>/dev/null | sed 's/^/[repro] invalid-memory-reference hits: /'
echo "[repro] --- tail ---"; tail -c 600 "$out"
