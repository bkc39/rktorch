#!/usr/bin/env bash
# run-all.sh <cpu|cuda|mps> [iterations]  env: REPRO_DEVICE REPRO_TIMEOUT REPRO_LOGDIR
set -u
set -o pipefail   # else the `| tee` at the end masks every arm's status
rc=0
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
  racket scripts/debug/exit-loop/preflight.rkt 2>&1 || { echo "PREFLIGHT FAILED"; rc=1; }
  echo
  echo "### piped-EOF shapes (EOF == Ctrl-D)"
  for f in scripts/debug/exit-loop/shapes/*.rkt; do
    ./scripts/debug/exit-loop/exit-loop.sh "$f" "$n" || rc=1
  done
  echo
  echo "### real-pty Ctrl-D"
  if python3 -c 'import pty' 2>/dev/null; then
    for m in idle printing busy intr spam; do
      python3 scripts/debug/exit-loop/pty-ctrl-d.py "$m" $(( n < 8 ? n : 8 )) 2>&1 || rc=1
    done
  else
    echo "SKIP: python3 pty unavailable"
  fi
  echo
  echo "### finalizer diagnostics at exit (needs this debug branch)"
  cat scripts/debug/exit-loop/shape-prelude.rkt \
      scripts/debug/exit-loop/shapes/s4-mixed.rkt \
    | RKTORCH_MEM_TRACE=1 timeout --kill-after=30 "$REPRO_TIMEOUT" racket -i \
        > "$REPRO_LOGDIR/.diag" 2>&1 || {
          _dc=$?
          echo "diagnostics arm FAILED: exit $_dc$([ "$_dc" = 124 ] && echo ' (timed out)')"
          rc=1
        }
  grep 'rktorch mem' "$REPRO_LOGDIR/.diag" > "$REPRO_LOGDIR/.trace" \
    || { echo "no trace line -- diagnostics arm FAILED"; rc=1; }
  if tr -d '\n\r; ' < "$REPRO_LOGDIR/.diag" | grep -qF 'btforcontext]'; then
    echo "diagnostics arm FAILED: the shape raised"; rc=1
  fi
  rm -f "$REPRO_LOGDIR/.diag"
  if [ -s "$REPRO_LOGDIR/.trace" ]; then
    cat "$REPRO_LOGDIR/.trace"
    _f=$(tail -1 "$REPRO_LOGDIR/.trace" \
           | grep -oE '\(failures \. [0-9]+\)' | head -1 | grep -oE '[0-9]+')
    if [ -n "$_f" ] && [ "$_f" -ne 0 ]; then
      echo "diagnostics arm FAILED: finalizer-failures=$_f"; rc=1
    fi
  fi
  rm -f "$REPRO_LOGDIR/.trace"
  echo
  echo "### done $(date)"
  exit "$rc"
} 2>&1 | tee "$report"
rc=$?

echo
echo "Report written to: $report"
echo "Failure transcripts (if any) in: $REPRO_LOGDIR"
exit "$rc"
