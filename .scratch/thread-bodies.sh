#!/usr/bin/env bash
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
gh api graphql -f query='query($n:Int!){ repository(owner:"bkc39",name:"rktorch"){ pullRequest(number:$n){ reviewThreads(first:50){ nodes{ id isResolved path line comments(first:5){ nodes{ author{login} body } } } } } } }' -F n="$1" --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved|not) | "\n########## \(.id) \(.path):\(.line)\n\(.comments.nodes[0].body)"'
