#!/usr/bin/env bash
# clang-format the two flagged files, then the format and tidy checks
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
nix develop .#ci -c clang-format -i cpp/src/torchrkt/autocast.cpp cpp/tests/torchrkt/dtype_test.cpp 2>&1 | grep -v "tree '/home\|^Staging\|^raco setup\|^Done. Lint\|full sweep"
echo "=== format+tidy"
nix build .#cpp-format .#cpp-tidy --no-link > .scratch/tidy-half.log 2>&1
echo "format+tidy exit: $?"
grep -v "tree '/home" .scratch/tidy-half.log | grep -E "error|warning" | head -20
echo "format-half done"
