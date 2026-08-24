#!/usr/bin/env bash
# Feed a shape file into `racket -i` on stdin (EOF == Ctrl-D), N times, and
# report how often the session ends badly.  Usage:
#   scripts/repro/exit-loop.sh scripts/repro/shapes/s1-churn-cpu.rkt 30
set -u
shape="${1:?usage: exit-loop.sh <shape.rkt> [iterations]}"
n="${2:-30}"
logdir="${REPRO_LOGDIR:-${TMPDIR:-/tmp}/rktorch-repro}"
secs="${REPRO_TIMEOUT:-180}"
mkdir -p "$logdir"
name=$(basename "$shape" .rkt)
bad=0
for i in $(seq 1 "$n"); do
  log="$logdir/$name-$i.log"
  timeout "$secs" racket -i < "$shape" > "$log" 2>&1
  code=$?
  if [ "$code" -ne 0 ] || grep -qE 'invalid memory reference|error display handler|error escape handler' "$log"; then
    bad=$((bad + 1))
    echo "FAIL iter=$i exit=$code size=$(wc -c < "$log") log=$log"
  else
    rm -f "$log"
  fi
done
echo "RESULT $name: $bad/$n failed"
