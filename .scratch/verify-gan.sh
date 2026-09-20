#!/usr/bin/env bash
# leg 4 verification: CPU suites, review, resyntax, docs; then the GPU twins when the card is idle
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep"
echo "=== cpu suites"
nix develop .#ci -c raco test examples/test/10-dcgan.rkt examples/test/11-vae.rkt examples/test/09-resnet.rkt torch/tests/ppm-test.rkt torch/tests/transforms-test.rkt torch/tests/nn-test.rkt > .scratch/gan-cpu.log 2>&1
echo "suites exit: $?"
grep -vE "$noise" .scratch/gan-cpu.log | grep -E "tests passed|test failures|FAILURE|non-zero exit|raised|rror" | tail -12
echo "=== raco review"
nix develop .#ci -c raco review examples/test/10-dcgan.rkt examples/test/11-vae.rkt torch/tests/python-cross-test.rkt 2>&1 | grep -vE "$noise" | grep -v "already defined\|should come before\|ctc-loss\|parens should be\|potentially unused require" | tail -12
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
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt examples/test/10-dcgan.rkt examples/test/11-vae.rkt > .scratch/gan-gpu.log 2>&1
echo "gpu suite exit: $?"
grep -E "tests passed|test failures|FAILURE|raised an exception|non-zero exit|python failed|python check" .scratch/gan-gpu.log | grep -v "tree '/home" | tail -8
echo "verify-gan done"
