#!/usr/bin/env bash
# each PR's own diff size against its base
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
for pr in 179 159 158 165 167 170 171; do
  gh pr view "$pr" --json number,additions,deletions,changedFiles,baseRefName,title \
    --jq '"#\(.number) +\(.additions)/-\(.deletions) in \(.changedFiles) files  -> \(.baseRefName)"'
done
