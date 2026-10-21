#!/usr/bin/env bash
#
# The L0 baseline (#254): tables A-G of docs/benchmarks/README.md, CPU then
# CUDA, each group started once the load is below BASELINE_MAX_LOAD, into
# docs/benchmarks/baseline-<date>.{jsonl,md}. Run it from the host shell; it
# enters the dev shells itself, one at a time.
#
#   scripts/bench/l0-baseline.sh             # all: cpu, cuda, summary
#   scripts/bench/l0-baseline.sh cpu         # one stage
#   BASELINE_SMOKE=1 BASELINE_OUT=/tmp/x/l0 scripts/bench/l0-baseline.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
self=$root/scripts/bench/l0-baseline.sh
date=${BASELINE_DATE:-$(date +%F)}
smoke=${BASELINE_SMOKE:-0}
if [ "$smoke" = 1 ] && [ -z "${BASELINE_OUT:-}" ]; then
  echo "l0-baseline: a smoke run needs BASELINE_OUT outside docs/" >&2
  exit 2
fi
out=${BASELINE_OUT:-$root/docs/benchmarks/baseline-$date}
export BASELINE_DATE=$date BASELINE_OUT=$out
cache=${BASELINE_CACHE:-$HOME/.racket/rktorch}
rounds=${BASELINE_ROUNDS:-3}
load=${BASELINE_MAX_LOAD:-4}
jsonl=$out.jsonl

if [ "$smoke" = 1 ]; then
  micro=(--scale 0.01 --rounds 1 --warmup 0)
  e2e=(--rounds 1)
  tiny=DATASET=fixture,EPOCHS=2,BATCH=128,BASE=4
  budgets=(1)
else
  micro=()
  e2e=(--rounds "$rounds")
  tiny=
  budgets=(0.01 0.05 0.2 1)
fi

variant() {
  local settings=$2
  if [ -n "$tiny" ]; then settings=${settings:+$settings,}$tiny; fi
  printf -- '--variant\n%s\n' "$1${settings:+:$settings}"
}

failed=0

bench() {
  racket scripts/bench/run.rkt --max-load "$load" --out "$jsonl" "$@" \
    || { echo "l0-baseline: run.rkt $* exited $?" >&2; failed=1; }
}

one_thread() {
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"
}

stage_cpu() {
  raco make -j 2 scripts/bench/run.rkt scripts/bench/summary.rkt
  bench "${micro[@]}" ops pytorch-ops
  one_thread bench "${micro[@]}" ops pytorch-ops
  one_thread bench "${micro[@]}" threads pytorch-threads
}

stage_cuda() {
  raco make -j 2 scripts/bench/run.rkt scripts/bench/summary.rkt
  bench "${micro[@]}" --device cuda ops pytorch-ops calls host-gpu
  local loader=()
  for v in pgm crop; do
    for w in 0 2 4; do
      mapfile -t -O "${#loader[@]}" loader < <(variant "$v-$w" "VARIANT=$v,WORKERS=$w")
    done
  done
  bench e2e "${e2e[@]}" --cache "$cache" --only 17-workers,py-17-workers "${loader[@]}"
  local headline=()
  mapfile -t headline < <(variant headline "")
  bench e2e "${e2e[@]}" --cache "$cache" --only 05,py-05,09,py-09 "${headline[@]}"
  local gc=()
  mapfile -t gc < <(variant troughs "")
  for b in "${budgets[@]}"; do
    mapfile -t -O "${#gc[@]}" gc \
      < <(variant "no-troughs-$b" "RKTORCH_BENCH_COLLECT_AT_TROUGHS=0,RKTORCH_BENCH_COLLECT_BUDGET=$b")
  done
  bench e2e "${e2e[@]}" --cache "$cache" --only 05,09 "${gc[@]}"
}

stage_summary() {
  racket scripts/bench/summary.rkt --title "Baseline $date (L0, #254)" "$jsonl" > "$out.md"
}

cd "$root"
mkdir -p "$(dirname "$out")"
case ${1:-all} in
  all)
    nix develop --max-jobs 1 --cores 4 -c "$self" cpu || failed=1
    env -u LD_LIBRARY_PATH nix develop --max-jobs 1 --cores 4 .#cuda -c "$self" cuda || failed=1
    nix develop --max-jobs 1 --cores 4 -c "$self" summary || failed=1
    ;;
  cpu) stage_cpu ;;
  cuda) stage_cuda ;;
  summary) stage_summary ;;
  *)
    echo "l0-baseline: unknown stage $1 (all, cpu, cuda or summary)" >&2
    exit 2
    ;;
esac
exit "$failed"
