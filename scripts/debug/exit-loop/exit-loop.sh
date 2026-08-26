#!/usr/bin/env bash
# exit-loop.sh <shape.rkt> [iterations]   env: REPRO_DEVICE REPRO_TIMEOUT REPRO_LOGDIR
set -u
set -o pipefail   # else `cat` failing on a missing shape is masked by racket
shape="${1:?usage: exit-loop.sh <shape.rkt> [iterations]}"
[ -r "$shape" ] || { echo "cannot read shape: $shape"; exit 1; }
prelude="$(dirname "${BASH_SOURCE[0]}")/shape-prelude.rkt"
[ -f "$prelude" ] || { echo "missing $prelude"; exit 1; }
n="${2:-20}"
case "$n" in ("" | *[!0-9]*) echo "iterations must be a positive integer: $n"; exit 2 ;; esac
[ "$n" -ge 1 ] || { echo "iterations must be >= 1: $n"; exit 2; }
dev="${REPRO_DEVICE:-cpu}"
logdir="${REPRO_LOGDIR:-${TMPDIR:-/tmp}/rktorch-repro}"
secs="${REPRO_TIMEOUT:-300}"
mkdir -p "$logdir"
name="$(basename "$shape" .rkt)-$dev"
bad=0

for i in $(seq 1 "$n"); do
  log="$logdir/$name-$i.log"
  cat "$prelude" "$shape" \
    | RKTORCH_MEM_TRACE=1 timeout --kill-after=30 "$secs" racket -i > "$log" 2>&1
  code=$?
  if grep -qE 'REPRO-DEVICE-(UNAVAILABLE|UNKNOWN)' "$log"; then
    echo "ABORT $name: device '$dev' is not available on this host"
    rm -f "$log"; exit 3
  fi
  fails=$(grep -a '\[rktorch mem\]' "$log" | tail -1 \
            | grep -oE '\(failures \. [0-9]+\)' | head -1 | grep -oE '[0-9]+')
  banner=0
  tr -d '\n\r; ' < "$log" | grep -qF 'btforcontext]' && banner=1
  if [ "$code" -ne 0 ] \
     || [ "$banner" = 1 ] \
     || [ -z "$fails" ] \
     || [ "$fails" -ne 0 ] \
     || grep -qE 'invalid memory reference|error display handler|error escape handler' "$log"; then
    bad=$((bad + 1))
    echo "FAIL iter=$i exit=$code finalizer-failures=${fails:-?} size=$(wc -c < "$log") log=$log"
  else
    rm -f "$log"
  fi
done
echo "RESULT $name: $bad/$n failed"
[ "$bad" -eq 0 ]
