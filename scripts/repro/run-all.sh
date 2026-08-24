#!/usr/bin/env bash
# Run the whole exit-loop sweep for one device and write a single report.
#
#   scripts/repro/run-all.sh cpu  [iterations]
#   scripts/repro/run-all.sh cuda [iterations]
#
# Must be run from the repo root, inside the dev shell (see the runbook).
set -u
dev="${1:-cpu}"
n="${2:-20}"
export REPRO_DEVICE="$dev"
export REPRO_LOGDIR="${REPRO_LOGDIR:-$PWD/repro-logs}"
export REPRO_TIMEOUT="${REPRO_TIMEOUT:-300}"
mkdir -p "$REPRO_LOGDIR"
report="$REPRO_LOGDIR/report-$dev.txt"

{
  echo "=============================================================="
  echo "rktorch exit-loop sweep   device=$dev  iterations=$n  $(date)"
  echo "host: $(uname -a)"
  echo "git:  $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"
  echo "=============================================================="
  echo
  echo "### preflight"
  racket scripts/repro/preflight.rkt 2>&1 || echo "PREFLIGHT FAILED"
  echo
  echo "### piped-EOF shapes (EOF == Ctrl-D)"
  for f in scripts/repro/shapes/*.rkt; do
    ./scripts/repro/exit-loop.sh "$f" "$n"
  done
  echo
  echo "### real-pty Ctrl-D"
  if python3 -c 'import pty' 2>/dev/null; then
    for m in idle printing busy intr spam; do
      python3 scripts/repro/pty-ctrl-d.py "$m" $(( n < 8 ? n : 8 )) 2>&1
    done
  else
    echo "SKIP: python3 pty unavailable"
  fi
  echo
  echo "### finalizer diagnostics at exit (needs this debug branch)"
  RKTORCH_MEM_TRACE=1 racket -i < scripts/repro/shapes/s4-mixed.rkt 2>&1 \
    | grep 'rktorch mem' || echo "no trace line -- are you on debug/exit-loop?"
  echo
  echo "### done $(date)"
} 2>&1 | tee "$report"

echo
echo "Report written to: $report"
echo "Failure transcripts (if any) in: $REPRO_LOGDIR"
