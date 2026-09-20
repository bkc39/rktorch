#!/usr/bin/env bash
# leg 3 verification: CPU suites, review, resyntax, docs; then the GPU twin and the example's accelerator arm
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep"
echo "=== cpu suites"
nix develop .#ci -c raco test examples/test/09-resnet.rkt torch/tests/nn-test.rkt torch/tests/nn-contract-test.rkt torch/tests/transforms-test.rkt examples/test/05-mnist.rkt torch/tests/convnet-smoke-test.rkt > .scratch/resnet-cpu.log 2>&1
echo "suites exit: $?"
grep -vE "$noise" .scratch/resnet-cpu.log | grep -E "tests passed|test failures|FAILURE|non-zero exit|raised|rror" | tail -12
echo "=== raco review"
nix develop .#ci -c raco review torch/vision/resnet.rkt torch/nn/conv.rkt examples/test/09-resnet.rkt torch/tests/python-cross-test.rkt torch/tests/nn-test.rkt 2>&1 | grep -vE "$noise" | grep -v "already defined\|should come before\|ctc-loss\|parens should be" | tail -12
echo "=== resyntax"
nix develop .#ci -c resyntax analyze --local-git-repository . origin/master 2>&1 | grep -vE "$noise" | grep -v "timed out\|analyzing\|^resyntax: skipping\|#<syntax" | tail -12
echo "=== docs"
nix develop .#ci -c scribble --dest .scratch/docs torch/scribblings/torch.scrbl > .scratch/docs.log 2>&1
echo "docs exit: $?"
grep -vE "$noise" .scratch/docs.log | grep -iv "cross references\|(dep " | head -6
echo "=== gpu wait"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  busy=$(pgrep -f "resyntax analyze" | wc -l)
  if [ "$used" -lt 2000 ] && [ "$busy" -eq 0 ]; then break; fi
  sleep 60
done
echo "gpu free (${used} MiB), cuda suites at $(date -u +%H:%M)"
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt examples/test/09-resnet.rkt > .scratch/resnet-gpu.log 2>&1
echo "gpu suite exit: $?"
grep -E "tests passed|test failures|FAILURE|raised an exception|non-zero exit|python failed|python check" .scratch/resnet-gpu.log | grep -v "tree '/home" | tail -8
echo "verify-resnet done"
