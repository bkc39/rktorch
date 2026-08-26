#!/usr/bin/env bash
# corrupt-restage.sh [zero|restage]       env: REPRO_DEVICE REPRO_LOGDIR
set -u
mode="${1:-zero}"
case "$mode" in zero|restage) ;; *) echo "unknown mode: $mode (want zero|restage)"; exit 2 ;; esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
case "$(uname -s)" in
  Darwin) _ext=dylib ;;
  *)      _ext=so ;;
esac
lib="$root/torch/native-libs/libtorchrkt.$_ext"
[ -f "$lib" ] || lib=""
[ -n "$lib" ] || { echo "no staged libtorchrkt in $root/torch/native-libs"; exit 1; }
dev="${REPRO_DEVICE:-cpu}"
out="${REPRO_LOGDIR:-${TMPDIR:-/tmp}}/corrupt-$mode-$dev.log"
backup=$(mktemp "${TMPDIR:-/tmp}/$(basename "$lib").XXXXXX.bak") || {
  echo "cannot create backup"; exit 1; }
mkdir -p "$(dirname "$out")"
cp -f "$lib" "$backup" && chmod u+w "$backup" || { echo "cannot populate backup"; exit 1; }
restore () {
  exec 3>&- 2>/dev/null || true
  if [ -n "${rkt:-}" ] && kill -0 "$rkt" 2>/dev/null; then
    kill -9 "$rkt" 2>/dev/null || true
    wait "$rkt" 2>/dev/null || true
  fi
  _tmp="$(dirname "$lib")/.restore.$$"
  if cp -f "$backup" "$_tmp" 2>/dev/null \
     && chmod 0555 "$_tmp" 2>/dev/null \
     && mv -f "$_tmp" "$lib" 2>/dev/null; then
    rm -f "$backup"
  else
    rm -f "$_tmp"
    echo "[repro] RESTORE FAILED -- staged shim may be corrupt; backup kept at $backup" >&2
  fi
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
chmod u+w "$lib"

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
printf '(unless (case "%s" [("cuda") (cuda-available?)] [("mps") (mps-available?)] [("cpu") #t] [else #f]) (printf "REPRO-DEVICE-BAD\\n") (exit 3))\n' "$dev" >&3
printf '(define D (case "%s" [("cuda") (cuda-device)] [("mps") (mps-device)] [else (cpu-device)]))\n' "$dev" >&3
sleep 3
if grep -q REPRO-DEVICE-BAD "$out" 2>/dev/null; then
  echo "[repro] ABORT: device '$dev' unusable on this host"; exit 3
fi
printf '(define held (with-default-device D (for/list ([_ (in-range 500)]) (randn 256 256))))\n' >&3
printf '(length held)\n' >&3
printf '(when (= (length held) 500) (printf "REPRO-SETUP-~a\\n" "READY"))\n' >&3
for _ in $(seq 1 60); do
  grep -q REPRO-SETUP-READY "$out" 2>/dev/null && break
  sleep 1
done
grep -q REPRO-SETUP-READY "$out" 2>/dev/null || {
  echo "[repro] ABORT: setup never completed; not corrupting"; exit 1; }

echo "[repro] overwriting mapped image in place (mode=$mode)"
case "$mode" in
  zero)    python3 -c "
import sys,os
p=sys.argv[1]; n=os.path.getsize(p)
f=open(p,'r+b'); f.seek(4096); f.write(b'\x00'*(n-4096)); f.flush(); os.fsync(f.fileno()); f.close()
print('  zeroed', n-4096, 'bytes')" "$lib" || { echo "[repro] ABORT: corruption failed"; exit 1; } ;;
  restage) before=$(stat -c%i "$lib" 2>/dev/null || stat -f%i "$lib")
           cp "$backup" "$lib" || { echo "[repro] ABORT: restage failed"; exit 1; }
           after=$(stat -c%i "$lib" 2>/dev/null || stat -f%i "$lib")
           echo "  re-copied identical bytes (inode $before -> $after$([ "$before" = "$after" ] && echo ', in place' || echo ' -- NOT in place, repro is inert'))" ;;
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

imr=$(grep -c 'invalid memory reference' "$out" || true)
casc=$(grep -cE 'error display handler|error escape handler' "$out" || true)
echo "[repro] exit=$code logbytes=$(wc -c < "$out")"
echo "[repro] invalid-memory-reference hits: $imr"
echo "[repro] cascade hits: $casc"
echo "[repro] --- tail ---"; tail -c 600 "$out"
banner=0
case "$(tr -d '\n\r; ' < "$out")" in *btforcontext]*) banner=1 ;; esac
if [ "$code" -ne 0 ] || [ "$imr" -ne 0 ] || [ "$casc" -ne 0 ] || [ "$banner" = 1 ]; then
  echo "[repro] RESULT: vector reproduced"
  exit 1
fi
echo "[repro] RESULT: no fault observed"
exit 0
