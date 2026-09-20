#!/usr/bin/env bash
# one line per PR of the vision stack: base <- head, draft, mergeable, non-passing checks, unresolved threads
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
for pr in 158 159 165 167 170 171; do
  meta=$(gh pr view "$pr" --json isDraft,mergeable,baseRefName,headRefName,reviewDecision --jq '"\(.baseRefName) <- \(.headRefName) draft=\(.isDraft) mergeable=\(.mergeable) decision=\(.reviewDecision)"')
  bad=$(gh pr checks "$pr" 2>/dev/null | awk -F'\t' '$2 != "pass" {printf "%s=%s ", $1, $2}')
  open=$(./.scratch/threads.sh "$pr" | grep -c "resolved=false")
  echo "#$pr $meta checks-not-passing=[${bad}] unresolved-threads=$open"
done
