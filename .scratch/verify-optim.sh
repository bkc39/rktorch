#!/usr/bin/env bash
# leg 2 verification: CPU suites, review, resyntax, docs; then the GPU twins when the card is idle
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep"
echo "=== cpu suites"
nix develop .#ci -c raco test torch/tests/nn-test.rkt torch/tests/scheduler-test.rkt torch/tests/to-test.rkt torch/tests/define-layer-test.rkt torch/tests/nn-contract-test.rkt torch/tests/diffusion-test.rkt torch/tests/convnet-smoke-test.rkt examples/test/04-mlp.rkt examples/test/05-mnist.rkt examples/test/06-gpt.rkt > .scratch/optim-cpu.log 2>&1
echo "suites exit: $?"
grep -vE "$noise" .scratch/optim-cpu.log | grep -E "tests passed|test failures|FAILURE|non-zero exit|raised" | tail -12
echo "=== raco review"
nix develop .#ci -c raco review torch/nn/optim.rkt torch/nn/scheduler.rkt torch/nn.rkt torch/tests/scheduler-test.rkt torch/tests/nn-test.rkt torch/tests/python-cross-test.rkt 2>&1 | grep -vE "$noise" | grep -v "already defined\|should come before\|ctc-loss\|parens should be\|review: ignore" | tail -20
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
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt torch/tests/to-test.rkt torch/tests/scheduler-test.rkt torch/tests/nn-test.rkt > .scratch/optim-gpu.log 2>&1
echo "gpu suite exit: $?"
grep -E "tests passed|test failures|FAILURE|raised an exception|non-zero exit|python failed|python check" .scratch/optim-gpu.log | grep -v "tree '/home" | tail -8
echo "verify-optim done"
