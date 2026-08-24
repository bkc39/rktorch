#!/usr/bin/env bash
# Feed a shape file into `racket -i` on stdin (EOF == Ctrl-D), N times, and
# report how often the session ends badly.  Linux and macOS.
#
#   REPRO_DEVICE=cuda scripts/repro/exit-loop.sh scripts/repro/shapes/s1-churn.rkt 20
#
# env: REPRO_DEVICE (cpu|cuda|mps, default cpu)   REPRO_TIMEOUT (sec, default 300)
#      REPRO_LOGDIR (default $TMPDIR/rktorch-repro)
set -u
shape="${1:?usage: exit-loop.sh <shape.rkt> [iterations]}"
n="${2:-20}"
dev="${REPRO_DEVICE:-cpu}"
logdir="${REPRO_LOGDIR:-${TMPDIR:-/tmp}/rktorch-repro}"
secs="${REPRO_TIMEOUT:-300}"
mkdir -p "$logdir"
name="$(basename "$shape" .rkt)-$dev"
bad=0

for i in $(seq 1 "$n"); do
  log="$logdir/$name-$i.log"
  timeout "$secs" racket -i < "$shape" > "$log" 2>&1
  code=$?
  if grep -q 'REPRO-DEVICE-UNAVAILABLE' "$log"; then
    echo "ABORT $name: device '$dev' is not available on this host"
    rm -f "$log"; exit 3
  fi
  # 124 = timeout(1) killed it; a hung REPL is exactly the symptom we are after
  if [ "$code" -ne 0 ] \
     || grep -qE 'invalid memory reference|error display handler|error escape handler' "$log"; then
    bad=$((bad + 1))
    echo "FAIL iter=$i exit=$code size=$(wc -c < "$log") log=$log"
  else
    rm -f "$log"
  fi
done
echo "RESULT $name: $bad/$n failed"
