#!/usr/bin/env bash
# merge each branch into its child so the whole stack carries master + the contract change
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
set -e
pairs="vision/classic-152:vision/half-152 vision/half-152:vision/optim-152 vision/optim-152:vision/resnet-152 vision/resnet-152:vision/gan-152"
for pair in $pairs; do
  parent="${pair%%:*}"; child="${pair##*:}"
  echo "=== $parent -> $child"
  git checkout -q "$child"
  if git merge --no-edit "$parent" > /tmp/merge-out 2>&1; then
    echo "  clean"
  else
    echo "  CONFLICTS:"
    git status --porcelain | grep "^UU\|^AA\|^DU\|^UD\|^AU\|^UA" || true
    exit 1
  fi
done
echo "cascade done"
