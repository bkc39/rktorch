#!/usr/bin/env bash
# the racket-dev loop on leg 0 after the master merge
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
git add -A > /dev/null 2>&1
echo "=== compile"
nix develop .#ci --command raco make torch/tests/*.rkt > .scratch/l0-make.log 2>&1
echo "make exit: $?"
echo "=== test"
nix develop .#ci --command raco test torch/ examples/test/ > .scratch/l0-test.log 2>&1
echo "test exit: $?"
grep -E "tests passed|test failures|FAILURE|non-zero exit" .scratch/l0-test.log | tail -5
echo "=== docs"
nix develop .#ci --command raco scribble --dest /tmp/doc-l0 \
  torch/scribblings/torch.scrbl > .scratch/l0-docs.log 2>&1
echo "docs exit: $?"
echo "verify-leg0-merge done"
