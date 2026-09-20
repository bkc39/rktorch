#!/usr/bin/env bash
# wait for the GPU (another session is training on it), then rerun the python-cross twin and the ppm suite on cuda
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  busy=$(pgrep -f "resyntax analyze" | wc -l)
  if [ "$used" -lt 2000 ] && [ "$busy" -eq 0 ]; then break; fi
  sleep 60
done
echo "gpu free (${used} MiB), cuda suites at $(date -u +%H:%M)"
PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:" nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt torch/tests/ppm-test.rkt > .scratch/ppm-gpu2.log 2>&1
echo "gpu suite exit: $?"
grep -E "tests passed|test failures|FAILURE|raised an exception|non-zero exit|python failed|python check" .scratch/ppm-gpu2.log | grep -v "tree '/home" | tail -6
echo "gpu-ppm done"
