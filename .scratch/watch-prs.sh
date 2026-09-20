#!/usr/bin/env bash
# emit settled CI checks and new review/comment counts for PRs 158 and 159; exit when both have settled checks and at least one review entry
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
declare -A prev seen
for pr in 158 159 165 167 170 171; do prev[$pr]=""; seen[$pr]=0; done
while true; do
  done_all=1
  for pr in 158 159 165 167 170 171; do
    s=$(gh pr checks "$pr" --json name,bucket 2>/dev/null)
    cur=$(jq -r '.[] | select(.bucket!="pending") | "\(.name): \(.bucket)"' <<<"$s" | sort | sed "s/^/PR $pr /")
    comm -13 <(echo "${prev[$pr]}") <(echo "$cur")
    prev[$pr]=$cur
    n=$(gh pr view "$pr" --json reviews,comments --jq '(.reviews|length)+(.comments|length)' 2>/dev/null || echo 0)
    if [ "$n" -gt "${seen[$pr]}" ]; then
      echo "PR $pr: $n review/comment entries now (was ${seen[$pr]})"
      seen[$pr]=$n
    fi
    if ! { [ -n "$s" ] && jq -e 'length>0 and all(.bucket!="pending")' <<<"$s" >/dev/null && [ "$n" -gt 0 ]; }; then
      done_all=0
    fi
  done
  if [ "$done_all" -eq 1 ]; then echo "both PRs: checks settled and reviews present"; break; fi
  sleep 60
done
