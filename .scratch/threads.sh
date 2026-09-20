#!/usr/bin/env bash
# list review threads on a PR: id, resolved, path:line, comment count; usage: threads.sh <pr>
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
gh api graphql -f query='query($n:Int!){ repository(owner:"bkc39",name:"rktorch"){ pullRequest(number:$n){ headRefOid reviewThreads(first:50){ nodes{ id isResolved path line comments(first:20){ totalCount nodes{ author{login} } } } } } } }' -F n="$1" --jq '.data.repository.pullRequest as $p | "head \($p.headRefOid)", ($p.reviewThreads.nodes[] | "\(.id) resolved=\(.isResolved) \(.path):\(.line) comments=\(.comments.totalCount) last=\(.comments.nodes[-1].author.login)")'
