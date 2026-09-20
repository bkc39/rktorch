#!/usr/bin/env bash
# format the touched C files, stage the new ones, build the library with gtests, then format/tidy checks
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
nix develop .#ci -c clang-format -i cpp/src/torchrkt/autocast.cpp cpp/include/torchrkt/c_api/autocast.h cpp/tests/torchrkt/autocast_test.cpp cpp/tests/torchrkt/dtype_test.cpp cpp/src/torchrkt/tensor.cpp cpp/src/torchrkt/creation.cpp cpp/tests/torchrkt/c_api_compile_test.c cpp/include/torchrkt/c_api/tensor.h cpp/include/torchrkt/c_api/creation.h cpp/src/torchrkt/detail/dtype.hpp 2>&1 | grep -v "tree '/home\|^Staging\|^raco setup\|^Done. Lint\|full sweep"
git add cpp/src/torchrkt/autocast.cpp cpp/include/torchrkt/c_api/autocast.h cpp/tests/torchrkt/autocast_test.cpp cpp/tests/torchrkt/dtype_test.cpp
echo "=== cpp build"
nix build .#cpp --no-link > .scratch/build-half.log 2>&1
echo "cpp exit: $?"
grep -v "tree '/home" .scratch/build-half.log | grep -E "error|FAILED|Failure|passed|failed" | head -30
echo "=== format+tidy"
nix build .#cpp-format .#cpp-tidy --no-link > .scratch/tidy-half.log 2>&1
echo "format+tidy exit: $?"
grep -v "tree '/home" .scratch/tidy-half.log | grep -E "error|warning" | head -20
echo "build-half done"
