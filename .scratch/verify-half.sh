#!/usr/bin/env bash
# leg 1 verification: CPU suites, review, resyntax, docs; then the GPU twins when the card is idle
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep"
echo "=== cpu suites"
nix develop .#ci -c raco test torch/tests/tensor-ops-test.rkt torch/tests/to-test.rkt torch/tests/nn-test.rkt torch/tests/autocast-test.rkt torch/tests/define-layer-test.rkt torch/tests/foreign-test.rkt torch/tests/device-test.rkt torch/tests/selection-ops-test.rkt torch/tests/bytes-ingestion-test.rkt torch/tests/diffusion-test.rkt torch/tests/transforms-test.rkt torch/tests/nn-contract-test.rkt torch/tests/convnet-smoke-test.rkt torch/tests/procedure-layer-test.rkt examples/test/04-mlp.rkt examples/test/05-mnist.rkt > .scratch/half-cpu.log 2>&1
echo "suites exit: $?"
grep -vE "$noise" .scratch/half-cpu.log | grep -E "tests passed|test failures|FAILURE|non-zero exit|raised" | tail -12
echo "=== raco review"
nix develop .#ci -c raco review torch/foreign/autocast.rkt torch/foreign/raw/autocast.rkt torch/foreign/ops.rkt torch/foreign/creation-ops.rkt torch/foreign/raw/tensor.rkt torch/foreign/raw/creation.rkt torch/foreign/structs.rkt torch/foreign/nn-promoted.rkt torch/nn/linear.rkt torch/nn/layer.rkt torch/nn/state-dict.rkt torch/tests/autocast-test.rkt torch/tests/tensor-ops-test.rkt torch/tests/to-test.rkt torch/tests/nn-test.rkt torch/tests/python-cross-test.rkt torch/tests/define-layer-test.rkt 2>&1 | grep -vE "$noise" | grep -v "already defined\|should come before\|ctc-loss\|parens should be\|ops.rkt:2\|layer.rkt:1[78]\|state-dict.rkt:.*total" | tail -20
echo "=== resyntax"
nix develop .#ci -c resyntax analyze --local-git-repository . origin/master 2>&1 | grep -vE "$noise" | grep -v "timed out\|analyzing\|^resyntax: skipping" | tail -12
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
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt torch/tests/autocast-test.rkt torch/tests/to-test.rkt torch/tests/tensor-ops-test.rkt torch/tests/nn-test.rkt torch/tests/generated-parity-test.rkt torch/tests/diffusion-test.rkt examples/test/08-diffusion.rkt > .scratch/half-gpu.log 2>&1
echo "gpu suite exit: $?"
grep -E "tests passed|test failures|FAILURE|raised an exception|non-zero exit|python failed|python check" .scratch/half-gpu.log | grep -v "tree '/home" | tail -8
echo "verify-half done"
