#!/usr/bin/env bash
# emit each settled CI check on PR 158 and each new review/comment count; exit when both are in
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
prev=""
seen=0
while true; do
  s=$(gh pr checks 158 --json name,bucket 2>/dev/null)
  cur=$(jq -r '.[] | select(.bucket!="pending") | "\(.name): \(.bucket)"' <<<"$s" | sort)
  comm -13 <(echo "$prev") <(echo "$cur")
  prev=$cur
  n=$(gh pr view 158 --json reviews,comments --jq '(.reviews|length)+(.comments|length)' 2>/dev/null || echo 0)
  if [ "$n" -gt "$seen" ]; then
    echo "PR 158: $n review/comment entries now (was $seen)"
    seen=$n
  fi
  if [ -n "$s" ] && jq -e 'length>0 and all(.bucket!="pending")' <<<"$s" >/dev/null && [ "$n" -gt 0 ]; then
    echo "PR 158: checks settled and reviews present"
    break
  fi
  sleep 60
done
