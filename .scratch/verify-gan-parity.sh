#!/usr/bin/env bash
# the python parity twins on the GPU shell, where python torch exists
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
export RKTORCH_CIFAR10_DIR="/home/bkc/.racket/rktorch/cifar10"
export RKTORCH_MNIST_DIR="/home/bkc/.racket/rktorch/mnist"
nix develop .#cuda -c raco test torch/tests/python-cross-test.rkt \
  > .scratch/gan-gpu3.log 2>&1
echo "exit $?"
grep -E "tests passed|test failures|FAILURE|python failed|non-zero" \
  .scratch/gan-gpu3.log | tail -8
