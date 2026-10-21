#!/usr/bin/env bash
#
#   scripts/platform/cuda-spike/sizes.sh <work-dir>   (the cut list, with sizes, after stage.sh)
set -euo pipefail

work=$(realpath "${1:?usage: sizes.sh <work-dir>}")
limit=$((2 * 1024 * 1024 * 1024))

printf '| package | zip (bytes) | zip (GiB) | margin under 2 GiB | unpacked (bytes) | zip/unpacked |\n'
printf '|---|---:|---:|---:|---:|---:|\n'
for zip in "$work"/zips/*.zip; do
  package=$(basename "$zip" .zip)
  zipped=$(stat -c %s "$zip")
  unpacked=$(find "$work/stage/$package" -type f -exec stat -c %s {} + | awk '{ s += $1 } END { print s }')
  awk -v p="$package" -v z="$zipped" -v u="$unpacked" -v l="$limit" 'BEGIN {
    printf "| %s | %d | %.3f | %.3f GiB (%.0f%%) | %d | %.2f |\n",
      p, z, z / 2^30, (l - z) / 2^30, 100 * (l - z) / l, u, z / u }'
done

for dir in "$work"/stage/*/; do
  printf '\n%s:\n' "$(basename "$dir")"
  find "$dir" -maxdepth 1 -type f -printf '  %12s  %f\n' | sort -k2
done
