#!/usr/bin/env bash
# exit once PR 159's checks have all settled, printing any failures
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
while true; do
  s=$(gh pr checks 159 --json name,bucket 2>/dev/null || true)
  if [ -n "$s" ] && jq -e 'length>0 and all(.bucket!="pending")' <<<"$s" >/dev/null 2>&1; then
    fails=$(jq -r '.[] | select(.bucket=="fail") | .name' <<<"$s" | paste -sd, -)
    if [ -n "$fails" ]; then echo "PR 159 SETTLED WITH FAILURES: $fails"; else echo "PR 159 settled: all checks pass"; fi
    exit 0
  fi
  sleep 60
done
