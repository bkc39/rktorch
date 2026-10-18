#!/usr/bin/env bash
#
#   scripts/platform/fetch.sh <download-dir> <file>...
set -euo pipefail

usage="usage: fetch.sh <download-dir> <file>..."
dl=${1:?$usage}
shift
[ "$#" -gt 0 ] || { echo "$usage" >&2; exit 2; }
sources=$(cd "$(dirname "$0")" && pwd)/sources.txt
mkdir -p "$dl"

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi | cut -d' ' -f1
}

for file in "$@"; do
  read -r sum url < <(awk -v file="$file" '$1 !~ /^#/ && $2 == file { print $1, $3 }' "$sources") ||
    { echo "fetch: $file is not pinned in $sources" >&2; exit 1; }
  if [ ! -f "$dl/$file" ]; then
    echo "fetch: $url" >&2
    curl -fsSL --retry 3 -o "$dl/$file.part" "$url"
    mv "$dl/$file.part" "$dl/$file"
  fi
  got=$(sha256 "$dl/$file")
  if [ "$got" != "$sum" ]; then
    echo "fetch: $file has sha256 $got; sources.txt pins $sum" >&2
    exit 1
  fi
  echo "$sum $file"
done
