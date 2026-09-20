#!/usr/bin/env bash
# how big each conflict in the dry-run merge actually is
cd /home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152 || exit 1
tree=8836b327b33805540817c77fe9635d2fc8a4d54d
for f in AGENTS.md torch/nn.rkt torch/nn/layer.rkt torch/nn/loss.rkt; do
  n=$(git show "$tree:$f" | grep -c '^<<<<<<<')
  lines=$(git show "$tree:$f" | awk '/^<<<<<<</{c=1} c{n++} /^>>>>>>>/{c=0} END{print n+0}')
  echo "$f: $n hunk(s), $lines conflicted lines"
done
