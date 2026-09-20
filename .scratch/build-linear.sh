#!/usr/bin/env bash
# regenerate with linear, format the touched C files, build with gtests, format/tidy
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
nix run .#codegen 2>&1 | grep -v "tree '/home" | tail -2
git add -A codegen cpp torch AGENTS.md
echo "=== clang-format"
nix develop .#ci -c clang-format -i cpp/tests/torchrkt/generated_tranche6_test.cpp cpp/tests/torchrkt/c_api_compile_test.c 2>&1 | grep -v "tree '/home\|^Staging\|^raco setup\|^Done. Lint\|full sweep"
echo "=== cpp build"
nix build .#cpp --no-link > .scratch/build-linear.log 2>&1
echo "cpp exit: $?"
grep -v "tree '/home" .scratch/build-linear.log | grep -E "error|FAILED|Failure" | head -20
echo "=== format+tidy"
nix build .#cpp-format .#cpp-tidy --no-link > .scratch/tidy-linear.log 2>&1
echo "format+tidy exit: $?"
grep -v "tree '/home" .scratch/tidy-linear.log | grep -E "error|warning" | head -20
echo "build-linear done"
