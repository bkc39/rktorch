#!/usr/bin/env bash
# #155 verification: ppm suite, review, resyntax, docs on the CPU; then python-cross on the GPU when idle
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep"
echo "=== ppm suite (cpu)"
nix develop .#ci -c raco test torch/tests/ppm-test.rkt > .scratch/ppm-cpu.log 2>&1
echo "suite exit: $?"
grep -vE "$noise" .scratch/ppm-cpu.log | grep -E "tests passed|test failures|FAILURE|error|exception" | tail -8
echo "=== raco review"
nix develop .#ci -c raco review torch/vision/ppm.rkt torch/tests/ppm-test.rkt torch/tests/python-cross-test.rkt 2>&1 | grep -vE "$noise" | tail -12
echo "=== resyntax"
nix develop .#ci -c resyntax analyze --local-git-repository . origin/master 2>&1 | grep -vE "$noise" | grep -v "timed out" | tail -8
echo "=== docs"
nix develop .#ci -c scribble --dest .scratch/docs torch/scribblings/torch.scrbl > .scratch/docs.log 2>&1
echo "docs exit: $?"
grep -vE "$noise" .scratch/docs.log | grep -iv "cross references\|(dep " | head -5
echo "=== gpu wait"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  busy=$(pgrep -f "resyntax analyze" | wc -l)
  if [ "$used" -lt 2000 ] && [ "$busy" -eq 0 ]; then break; fi
  sleep 30
done
echo "gpu free (${used} MiB), python-cross + ppm on cuda at $(date -u +%H:%M)"
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt torch/tests/ppm-test.rkt > .scratch/ppm-gpu.log 2>&1
echo "gpu suite exit: $?"
grep -E "tests passed|test failures|FAILURE|ERROR|raised an exception|non-zero exit|python failed|python check|skipped" .scratch/ppm-gpu.log | grep -v "tree '/home" | tail -8
echo "verify-ppm done"
