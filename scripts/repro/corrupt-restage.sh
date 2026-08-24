#!/usr/bin/env bash
# Issue #72's proposed vector: overwrite the mapped libtorchrkt in place under a
# live REPL that holds tensors, then touch native code and send EOF (Ctrl-D).
# Restores the library on the way out.  Linux and macOS.
#
#   scripts/repro/corrupt-restage.sh [zero|restage]
#     zero    (default) zero the whole image past the ELF/Mach header
#     restage re-copy identical bytes in place -- what `nix run .#copy-native-libs` does
#
# On darwin this was measured INERT: the ~195 KB shim is fully resident after
# load, so writes to the file never reach the mapped pages.  On Linux with
# demand paging the outcome may differ -- that is the point of running it there.
set -u
mode="${1:-zero}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
lib="$(ls "$root"/torch/native-libs/libtorchrkt.so "$root"/torch/native-libs/libtorchrkt.dylib 2>/dev/null | head -1)"
[ -n "$lib" ] || { echo "no staged libtorchrkt in $root/torch/native-libs"; exit 1; }
dev="${REPRO_DEVICE:-cpu}"
out="${REPRO_LOGDIR:-${TMPDIR:-/tmp}}/corrupt-$mode-$dev.log"
backup="${TMPDIR:-/tmp}/$(basename "$lib").bak"
mkdir -p "$(dirname "$out")"
cp "$lib" "$backup"

snapshot () {  # $1 = pid
  if command -v sample >/dev/null 2>&1; then sample "$1" 3 -f "${out%.log}-sample.txt" 2>/dev/null
  elif command -v eu-stack >/dev/null 2>&1; then eu-stack -p "$1" > "${out%.log}-stack.txt" 2>&1
  elif command -v gdb >/dev/null 2>&1; then gdb -p "$1" -batch -ex 'thread apply all bt' > "${out%.log}-stack.txt" 2>&1
  fi
  [ -r "/proc/$1/status" ] && grep -E 'VmRSS|Threads' "/proc/$1/status"
  ps -o rss= -p "$1" 2>/dev/null
}

fifo=$(mktemp -u); mkfifo "$fifo"
racket -i < "$fifo" > "$out" 2>&1 &
rkt=$!
exec 3> "$fifo"
echo "[repro] pid=$rkt device=$dev lib=$lib log=$out"

printf '(require torch)\n' >&3
printf '(define D (case "%s" [("cuda") (cuda-device)] [("mps") (mps-device)] [else (cpu-device)]))\n' "$dev" >&3
printf '(define held (with-default-device D (for/list ([_ (in-range 500)]) (randn 256 256))))\n' >&3
printf '(length held)\n' >&3
sleep 8

echo "[repro] overwriting mapped image in place (mode=$mode)"
case "$mode" in
  zero)    python3 -c "
import sys,os
p=sys.argv[1]; n=os.path.getsize(p)
f=open(p,'r+b'); f.seek(4096); f.write(b'\x00'*(n-4096)); f.flush(); os.fsync(f.fileno()); f.close()
print('  zeroed', n-4096, 'bytes')" "$lib" ;;
  restage) cp "$backup" "$lib" && echo "  re-copied identical bytes" ;;
esac
sleep 2

echo "[repro] exercising cold native code paths"
for e in '(matmul (randn 32 32) (randn 32 32))' '(sum (randn 16 16))' \
         '(reshape (arange 0 64) 8 8)' '(exp (randn 4 4))' '(eye 5)' \
         '(cat (list (ones 2 2) (zeros 2 2)) 0)' '(car held)'; do
  printf '(with-default-device D %s)\n' "$e" >&3
done
sleep 6

echo "[repro] sending EOF (Ctrl-D)"
exec 3>&-
for _ in $(seq 1 40); do kill -0 "$rkt" 2>/dev/null || break; sleep 1; done
if kill -0 "$rkt" 2>/dev/null; then
  echo "[repro] STILL ALIVE after 40s -- snapshotting"; snapshot "$rkt"
  echo "[repro] SIGTERM"; kill -TERM "$rkt" 2>/dev/null; sleep 5
  if kill -0 "$rkt" 2>/dev/null; then echo "[repro] TERM IGNORED -> SIGKILL"; kill -9 "$rkt"; fi
fi
wait "$rkt" 2>/dev/null; code=$?
rm -f "$fifo"
cp "$backup" "$lib"; chmod u+w "$lib" 2>/dev/null || true

echo "[repro] exit=$code logbytes=$(wc -c < "$out")"
echo "[repro] invalid-memory-reference hits: $(grep -c 'invalid memory reference' "$out" || true)"
echo "[repro] cascade hits: $(grep -cE 'error display handler|error escape handler' "$out" || true)"
echo "[repro] --- tail ---"; tail -c 600 "$out"
