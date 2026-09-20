#!/usr/bin/env bash
# count unresolved review threads on every PR in the stack
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
for pr in 158 165 167 170 171 179; do
  n=$(gh api graphql -f query='query($n:Int!){ repository(owner:"bkc39",name:"rktorch"){ pullRequest(number:$n){ reviewThreads(first:60){ nodes{ isResolved } } } } }' -F n="$pr" --jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved|not)] | length')
  head=$(gh pr view "$pr" --json headRefOid --jq '.headRefOid' | cut -c1-7)
  echo "PR $pr: $n unresolved, head $head"
done
