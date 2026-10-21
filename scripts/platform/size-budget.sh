#!/usr/bin/env bash
#
#   scripts/platform/size-budget.sh <platform-zip>
set -euo pipefail

zip=${1:?usage: size-budget.sh <platform-zip>}
name=$(basename "$zip" .zip)
case $name in
  torch-x86_64-linux) budget=125 ;;
  torch-aarch64-macosx) budget=100 ;;
  *) echo "size-budget: no budget for $name" >&2; exit 2 ;;
esac

bytes=$(wc -c <"$zip")
megabytes=$(awk -v b="$bytes" 'BEGIN { printf "%.1f", b / 1e6 }')
echo "$name.zip: $megabytes MB, budget $budget MB"
if [ "$bytes" -gt $((budget * 1000000)) ]; then
  echo "size-budget: $name.zip is over its budget of $budget MB" >&2
  exit 1
fi
