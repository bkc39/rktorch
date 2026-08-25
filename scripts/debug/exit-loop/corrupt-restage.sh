#!/usr/bin/env bash
# Issue #72's proposed vector: overwrite the mapped libtorchrkt in place under a
# live REPL that holds tensors, then touch native code and send EOF (Ctrl-D).
# Restores the library on the way out.  Linux and macOS.
#
#   scripts/debug/exit-loop/corrupt-restage.sh [zero|restage]
#     zero    (default) zero the whole image past the ELF/Mach header
#     restage re-copy identical bytes in place -- what `nix run .#copy-native-libs` does
#
# On darwin this was measured INERT: the ~195 KB shim is fully resident after
# load, so writes to the file never reach the mapped pages.  On Linux with
# demand paging the outcome may differ -- that is the point of running it there.
set -u
mode="${1:-zero}"
case "$mode" in zero|restage) ;; *) echo "unknown mode: $mode (want zero|restage)"; exit 2 ;; esac
# Resolve the root with git: a fixed `..` count breaks whenever this file moves.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
# Pick by platform: `ls` would sort a stale .dylib ahead of the .so that
# Racket actually maps on Linux.
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
# Restore even on Ctrl-C or an early exit: leaving a zeroed shim staged
# breaks every later run in this worktree.
restore () {
  # Stop the child FIRST.  Restoring while it still has the library mapped
  # is another in-place write under a live process -- the corruption this
  # script exists to study -- and would leave the EOF-hung child orphaned.
  exec 3>&- 2>/dev/null || true
  if [ -n "${rkt:-}" ] && kill -0 "$rkt" 2>/dev/null; then
    kill -9 "$rkt" 2>/dev/null || true
    wait "$rkt" 2>/dev/null || true
  fi
  chmod u+w "$lib" 2>/dev/null || true
  cp -f "$backup" "$lib" 2>/dev/null || true
  chmod 0555 "$lib" 2>/dev/null || true
  rm -f "$backup"
}
trap restore EXIT
# INT/TERM must TERMINATE, not just restore and fall through: a signal
# arriving before the corruption step would otherwise let execution reach
# the destructive write with the backup already removed.
trap 'exit 130' INT
trap 'exit 143' TERM
# Staging leaves the shim 0555.  Without this, `cp` fails (or `cp -f`
# unlinks and creates a NEW inode) and the live mapping is untouched --
# the repro would quietly test nothing.
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
# Refuse to corrupt the shim for a device this host cannot run.
if grep -q REPRO-DEVICE-BAD "$out" 2>/dev/null; then
  echo "[repro] ABORT: device '$dev' unusable on this host"; exit 3
fi
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
  restage) before=$(stat -c%i "$lib" 2>/dev/null || stat -f%i "$lib")
           cp "$backup" "$lib"
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
cp "$backup" "$lib"; chmod u+w "$lib" 2>/dev/null || true

echo "[repro] exit=$code logbytes=$(wc -c < "$out")"
echo "[repro] invalid-memory-reference hits: $(grep -c 'invalid memory reference' "$out" || true)"
echo "[repro] cascade hits: $(grep -cE 'error display handler|error escape handler' "$out" || true)"
echo "[repro] --- tail ---"; tail -c 600 "$out"
