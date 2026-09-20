#!/usr/bin/env bash
# wait for the GPU to be idle and no resyntax/ci shell to be staging here, then run the GPU suites
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  busy=$(pgrep -f "resyntax analyze" | wc -l)
  if [ "$used" -lt 2000 ] && [ "$busy" -eq 0 ]; then break; fi
  sleep 30
done
echo "gpu free (${used} MiB used), starting suites at $(date -u +%H:%M)"
PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:" nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt torch/tests/generated-parity-test.rkt torch/tests/transforms-test.rkt torch/tests/nn-test.rkt torch/tests/tensor-ops-test.rkt torch/tests/to-test.rkt > .scratch/gpu-leg0.log 2>&1
echo "suite exit: $?"
grep -E "tests passed|test failures|FAILURE|ERROR|raised an exception|non-zero exit|python failed|skipped" .scratch/gpu-leg0.log | grep -v "tree '/home" | tail -8
echo "gpu suite done at $(date -u +%H:%M)"
