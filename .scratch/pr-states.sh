#!/usr/bin/env bash
# exact state of each PR in the two arcs
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
for n in 158 159 160 163 164 165 166 167 170 171 179; do
  gh pr view "$n" --json number,state,baseRefName,headRefName \
    --jq '"#\(.number) \(.state) \(.headRefName) -> \(.baseRefName)"' 2>/dev/null
done
