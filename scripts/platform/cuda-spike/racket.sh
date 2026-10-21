#!/usr/bin/env bash
#
#   scripts/platform/cuda-spike/racket.sh <work-dir>   (stock racket-lang.org Racket 9.1 in <work-dir>/racket)
set -euo pipefail

work=$(realpath -m "${1:?usage: racket.sh <work-dir>}")
here=$(cd "$(dirname "$0")" && pwd)
installer=racket-9.1-x86_64-linux-buster-cs.sh
bash "$here/../fetch.sh" "$work/dl" "$installer" >/dev/null
rm -rf "$work/racket"
sh "$work/dl/$installer" --in-place --dest "$work/racket" >"$work/racket-install.log" 2>&1 ||
  { tail -20 "$work/racket-install.log"; exit 1; }
"$work/racket/bin/racket" --version
