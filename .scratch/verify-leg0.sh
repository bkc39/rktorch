#!/usr/bin/env bash
# leg 0 verification: format/tidy, CPU suites, review, resyntax, docs
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
echo "=== format+tidy"
nix build .#cpp-format .#cpp-tidy --no-link 2>&1 | grep -v "tree '/home" | tail -8
echo "format+tidy exit: ${PIPESTATUS[0]}"
echo "=== racket suites"
nix develop .#ci -c raco test \
  torch/tests/nn-test.rkt torch/tests/nn-contract-test.rkt \
  torch/tests/tensor-ops-test.rkt torch/tests/transforms-test.rkt \
  torch/tests/generated-parity-test.rkt torch/tests/python-cross-test.rkt \
  torch/tests/define-layer-test.rkt torch/tests/foreign-test.rkt \
  torch/tests/to-test.rkt > .scratch/leg0-suites.log 2>&1
echo "suites exit: $?"
grep -E "tests passed|test failures|FAILURE|ERROR|raised an exception|non-zero exit|python failed|skipped" .scratch/leg0-suites.log | grep -v "tree '/home" | tail -12
echo "=== raco review"
nix develop .#ci -c raco review torch/foreign/nn-promoted.rkt torch/nn/loss.rkt torch/nn/batch-norm.rkt torch/vision/transforms.rkt torch/tests/transforms-test.rkt torch/tests/nn-test.rkt torch/tests/tensor-ops-test.rkt torch/tests/generated-parity-test.rkt torch/tests/python-cross-test.rkt torch/tests/nn-contract-test.rkt 2>&1 | grep -v "tree '/home" | tail -20
echo "review exit: ${PIPESTATUS[0]}"
echo "=== resyntax"
nix develop .#ci -c resyntax analyze --local-git-repository . origin/master 2>&1 | grep -v "tree '/home" | tail -12
echo "=== docs"
nix develop .#ci -c scribble --dest .scratch/docs torch/scribblings/torch.scrbl 2>&1 | grep -v "tree '/home" | tail -6
echo "docs exit: ${PIPESTATUS[0]}"
echo "verify done"
