#!/usr/bin/env bash
# leg 0 re-verification after the tensor? import fix: docs error, transforms suite, resyntax
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
noise="tree '/home|^Staging|^raco setup|^Done. Lint|full sweep"
echo "=== docs"
nix develop .#ci -c scribble --dest .scratch/docs torch/scribblings/torch.scrbl > .scratch/docs.log 2>&1
echo "docs exit: $?"
grep -vE "$noise" .scratch/docs.log | head -12
echo "=== transforms suite"
nix develop .#ci -c raco test torch/tests/transforms-test.rkt > .scratch/transforms.log 2>&1
echo "suite exit: $?"
grep -vE "$noise" .scratch/transforms.log | tail -15
echo "=== resyntax"
nix develop .#ci -c resyntax analyze --local-git-repository . origin/master > .scratch/resyntax.log 2>&1
grep -vE "$noise" .scratch/resyntax.log | tail -10
echo "verify-b done"
