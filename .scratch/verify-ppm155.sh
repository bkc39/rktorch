#!/usr/bin/env bash
# the whole CPU suite on vision/ppm-155 after the fill-value/c export
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
export PLTCOLLECTS="/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152:"
nix develop .#ci -c raco test torch/ examples/test/ > .scratch/ppm155-cpu.log 2>&1
echo "cpu exit: $?"
grep -E "tests passed|test failures|FAILURE|non-zero exit" .scratch/ppm155-cpu.log | tail -6
echo "verify-ppm155 done"
