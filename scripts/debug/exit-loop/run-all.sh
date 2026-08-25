#!/usr/bin/env bash
# Run the whole exit-loop sweep for one device and write a single report.
#
#   scripts/debug/exit-loop/run-all.sh cpu  [iterations]
#   scripts/debug/exit-loop/run-all.sh cuda [iterations]
#
# Must be run from the repo root, inside the dev shell:
#   nix develop        --command ./scripts/debug/exit-loop/run-all.sh cpu  20
#   nix develop .#cuda --command ./scripts/debug/exit-loop/run-all.sh cuda 20
# Stage a matching native lib first (`nix run .#copy-native-libs`); a shim
# that does not match the bytecode is itself a fault source.
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
    | RKTORCH_MEM_TRACE=1 racket -i 2>&1 \
    | grep 'rktorch mem' || { echo "no trace line -- diagnostics arm FAILED"; rc=1; }
  echo
  echo "### done $(date)"
  # The group runs in a subshell (left side of the pipe), so this status is the
  # only way its result reaches the caller.  A sweep whose arms aborted --
  # device unavailable, missing shim -- must not look green: that is how a run
  # that tested nothing gets believed.
  exit "$rc"
} 2>&1 | tee "$report"
rc=$?

echo
echo "Report written to: $report"
echo "Failure transcripts (if any) in: $REPRO_LOGDIR"
exit "$rc"
