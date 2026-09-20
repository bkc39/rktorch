#!/usr/bin/env bash
# emit each settled CI check and each new review entry for the vision stack and the audio fix
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
declare -A prev seen
for pr in 158 159 165 167 170 171 179; do prev[$pr]=""; seen[$pr]=0; done
while true; do
  for pr in 158 159 165 167 170 171 179; do
    s=$(gh pr checks "$pr" --json name,bucket 2>/dev/null || true)
    cur=$(jq -r '.[] | select(.bucket=="fail") | "\(.name): FAIL"' <<<"$s" 2>/dev/null | sort | sed "s/^/PR $pr /")
    comm -13 <(echo "${prev[$pr]}") <(echo "$cur") 2>/dev/null
    prev[$pr]=$cur
    n=$(gh pr view "$pr" --json reviews,comments --jq '(.reviews|length)+(.comments|length)' 2>/dev/null || echo 0)
    if [ "$n" -gt "${seen[$pr]}" ]; then
      [ "${seen[$pr]}" -gt 0 ] && echo "PR $pr: new review activity ($n entries)"
      seen[$pr]=$n
    fi
  done
  sleep 90
done
