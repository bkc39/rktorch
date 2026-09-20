#!/usr/bin/env bash
# a one-line CI summary per PR
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
for pr in 158 165 167 170 171 179; do
  s=$(gh pr checks "$pr" --json name,bucket 2>/dev/null)
  pass=$(jq -r '[.[] | select(.bucket=="pass")] | length' <<<"$s")
  fail=$(jq -r '[.[] | select(.bucket=="fail")] | length' <<<"$s")
  pend=$(jq -r '[.[] | select(.bucket=="pending")] | length' <<<"$s")
  names=$(jq -r '.[] | select(.bucket=="fail") | .name' <<<"$s" | paste -sd, -)
  echo "PR $pr: pass=$pass fail=$fail pending=$pend ${names:+FAILING: $names}"
done
