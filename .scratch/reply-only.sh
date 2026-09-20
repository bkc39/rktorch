#!/usr/bin/env bash
# reply on a review thread without resolving it; usage: reply-only.sh <threadId> <bodyfile>
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
set -e
tid="$1"; body="$(cat "$2")"
gh api graphql -f query='mutation($t:ID!,$b:String!){ addPullRequestReviewThreadReply(input:{pullRequestReviewThreadId:$t, body:$b}){ comment{ id } } }' -f t="$tid" -f b="$body" --jq '.data.addPullRequestReviewThreadReply.comment.id'
