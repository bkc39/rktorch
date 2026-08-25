#!/usr/bin/env bash
# Feed a shape file into `racket -i` on stdin (EOF == Ctrl-D), N times, and
# report how often the session ends badly.  Linux and macOS.
#
#   REPRO_DEVICE=cuda scripts/debug/exit-loop/exit-loop.sh scripts/debug/exit-loop/shapes/s1-churn.rkt 20
#
# env: REPRO_DEVICE (cpu|cuda|mps, default cpu)   REPRO_TIMEOUT (sec, default 300)
#      REPRO_LOGDIR (default $TMPDIR/rktorch-repro)
set -u
set -o pipefail   # else `cat` failing on a missing shape is masked by racket
shape="${1:?usage: exit-loop.sh <shape.rkt> [iterations]}"
[ -r "$shape" ] || { echo "cannot read shape: $shape"; exit 1; }
prelude="$(dirname "${BASH_SOURCE[0]}")/shape-prelude.rkt"
[ -f "$prelude" ] || { echo "missing $prelude"; exit 1; }
n="${2:-20}"
dev="${REPRO_DEVICE:-cpu}"
logdir="${REPRO_LOGDIR:-${TMPDIR:-/tmp}/rktorch-repro}"
secs="${REPRO_TIMEOUT:-300}"
mkdir -p "$logdir"
name="$(basename "$shape" .rkt)-$dev"
bad=0

for i in $(seq 1 "$n"); do
  log="$logdir/$name-$i.log"
  # --kill-after: the failure under test is a REPL that IGNORES SIGTERM, and
  # plain `timeout` only sends TERM -- without this the sweep hangs forever.
  # prelude + shape on stdin: the device selection is shared, not copied
  # into all seven shapes.  EOF at the end is the Ctrl-D under test.
  cat "$prelude" "$shape" \
    | RKTORCH_MEM_TRACE=1 timeout --kill-after=30 "$secs" racket -i > "$log" 2>&1
  code=$?
  if grep -qE 'REPRO-DEVICE-(UNAVAILABLE|UNKNOWN)' "$log"; then
    echo "ABORT $name: device '$dev' is not available on this host"
    rm -f "$log"; exit 3
  fi
  # 124 = timeout(1) killed it; a hung REPL is exactly the symptom we are after
  # `racket -i` keeps going after an error and still exits 0 on EOF, so a shape
  # that errored would score as a pass; xrepl's banner is the only signal.
  # A swallowed native free is the #72 signature and leaves exit 0 with none of
  # the phrases below, so read the exit-time diagnostic rather than trusting them.
  fails=$(grep -oE '\(failures \. [0-9]+\)' "$log" | grep -oE '[0-9]+' | tail -1)
  # xrepl wraps long messages, splitting the banner across lines with a `;`
  # continuation, so match it with the line structure removed.
  banner=0
  case "$(tr -d '\n\r; ' < "$log")" in *btforcontext]*) banner=1 ;; esac
  if [ "$code" -ne 0 ] \
     || [ "$banner" = 1 ] \
     || { [ -n "$fails" ] && [ "$fails" -ne 0 ]; } \
     || grep -qE 'invalid memory reference|error display handler|error escape handler' "$log"; then
    bad=$((bad + 1))
    echo "FAIL iter=$i exit=$code finalizer-failures=${fails:-?} size=$(wc -c < "$log") log=$log"
  else
    rm -f "$log"
  fi
done
echo "RESULT $name: $bad/$n failed"
# Nonzero so run-all.sh aggregates real failures, not only ABORTs.
[ "$bad" -eq 0 ]
