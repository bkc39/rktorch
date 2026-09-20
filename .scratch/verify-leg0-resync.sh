#!/usr/bin/env bash
# leg 0 after the forward contracts and the master merge: cpp gates, CPU suites, linters, docs, then GPU
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep|bytecode cache"
echo "=== codegen drift"
nix run .#codegen > /dev/null 2>&1
echo "drift files: $(git status --porcelain | grep -v scratch | wc -l)"
echo "=== cpp gates"
nix build .#cpp .#cpp-format .#cpp-tidy --no-link > .scratch/cpp.log 2>&1
echo "cpp exit: $?"
grep -vE "$noise" .scratch/cpp.log | grep -E "error|warning|FAILED" | head -10
echo "=== cpu suites"
nix develop .#ci -c raco test torch/tests/nn-test.rkt torch/tests/nn-contract-test.rkt torch/tests/define-layer-test.rkt torch/tests/forward-arity-test.rkt torch/tests/tensor-ops-test.rkt torch/tests/transforms-test.rkt torch/tests/diffusion-test.rkt torch/tests/generated-parity-test.rkt torch/tests/foreign-test.rkt torch/tests/to-test.rkt torch/tests/procedure-layer-test.rkt examples/test/04-mlp.rkt examples/test/05-mnist.rkt > .scratch/cpu.log 2>&1
echo "suites exit: $?"
grep -vE "$noise" .scratch/cpu.log | grep -E "tests passed|test failures|FAILURE|non-zero exit|raised" | tail -8
echo "=== raco review"
nix develop .#ci -c raco review torch/nn/layer.rkt torch/nn/batch-norm.rkt torch/private/definer.rkt torch/foreign/contracts.rkt torch/tests/nn-contract-test.rkt torch/tests/nn-test.rkt 2>&1 | grep -vE "$noise" | grep -v "already defined\|should come before\|never used" | tail -10
echo "=== resyntax"
nix develop .#ci -c resyntax analyze --local-git-repository . origin/master 2>&1 | grep -vE "$noise" | grep -v "timed out\|analyzing\|^resyntax: skipping\|#<syntax" | tail -8
echo "=== docs"
nix develop .#ci -c scribble --dest .scratch/docs torch/scribblings/torch.scrbl > .scratch/docs.log 2>&1
echo "docs exit: $?"
grep -vE "$noise" .scratch/docs.log | grep -iv "cross references\|(dep " | head -5
echo "=== gpu wait"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  busy=$(pgrep -f "resyntax analyze" | wc -l)
  if [ "$used" -lt 2000 ] && [ "$busy" -eq 0 ]; then break; fi
  sleep 60
done
echo "gpu free (${used} MiB) at $(date -u +%H:%M)"
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt torch/tests/generated-parity-test.rkt torch/tests/nn-test.rkt torch/tests/diffusion-test.rkt > .scratch/gpu.log 2>&1
echo "gpu exit: $?"
grep -E "tests passed|test failures|FAILURE|raised an exception|non-zero exit|python failed|python check" .scratch/gpu.log | grep -v "tree '/home" | tail -6
echo "verify-resync done"
