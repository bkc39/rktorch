#!/usr/bin/env bash
# the top of the stack after the cascade: cpp gates, the whole CPU suite, then GPU
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep|bytecode cache"
echo "=== drift"
nix run .#codegen > /dev/null 2>&1
echo "drift files: $(git status --porcelain | grep -v scratch | wc -l)"
echo "=== cpp"
nix build .#cpp .#cpp-format .#cpp-tidy --no-link > .scratch/top-cpp.log 2>&1
echo "cpp exit: $?"
echo "=== cpu: the whole torch suite plus the examples"
nix develop .#ci -c raco test torch/ examples/test/ > .scratch/top-cpu.log 2>&1
echo "cpu exit: $?"
grep -vE "$noise" .scratch/top-cpu.log | grep -E "tests passed|test failures|FAILURE|non-zero exit|raised|rror" | tail -10
echo "=== gpu wait"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 2000 ] && [ "$(pgrep -f 'resyntax analyze' | wc -l)" -eq 0 ]; then break; fi
  sleep 60
done
echo "gpu free (${used} MiB)"
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt torch/tests/generated-parity-test.rkt torch/tests/autocast-test.rkt examples/test/09-resnet.rkt > .scratch/top-gpu.log 2>&1
echo "gpu exit: $?"
grep -E "tests passed|test failures|FAILURE|python failed|python check|non-zero" .scratch/top-gpu.log | grep -v "tree '/home" | tail -6
echo "verify-top done"
