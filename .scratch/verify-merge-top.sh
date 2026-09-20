#!/usr/bin/env bash
# the full racket-dev loop on the top of the stack after the master merge
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
git add -A > /dev/null 2>&1
echo "=== drift"
nix run .#codegen > /dev/null 2>&1
echo "drift files: $(git status --porcelain | grep -v scratch | wc -l)"
echo "=== compile"
nix develop .#ci --command raco make torch/tests/*.rkt > .scratch/top-make.log 2>&1
echo "make exit: $?"
echo "=== test"
nix develop .#ci --command raco test torch/ examples/test/ > .scratch/top2-test.log 2>&1
echo "test exit: $?"
grep -E "tests passed|test failures|FAILURE|non-zero exit" .scratch/top2-test.log | tail -4
echo "=== coverage"
nix develop .#ci --command racket scripts/coverage.rkt --changed \
  > .scratch/top-cov.log 2>&1
echo "coverage exit: $?"
tail -4 .scratch/top-cov.log
echo "=== resyntax"
nix develop .#ci --command resyntax analyze --local-git-repository . \
  origin/master --analyzer-timeout 30000 > .scratch/top-resyntax.log 2>&1
echo "resyntax exit: $?"
grep -cE "^\[" .scratch/top-resyntax.log 2>/dev/null
echo "verify-merge-top done"
