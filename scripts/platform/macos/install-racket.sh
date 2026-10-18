#!/usr/bin/env bash
#
#   scripts/platform/macos/install-racket.sh <version> <dest> <download-dir>
set -euo pipefail

usage="usage: install-racket.sh <version> <dest> <download-dir>"
version=${1:?$usage}
dest=${2:?$usage}
dl=${3:?$usage}
here=$(cd "$(dirname "$0")" && pwd)
dmg=racket-$version-aarch64-macosx-cs.dmg

"$here/../fetch.sh" "$dl" "$dmg"
mnt=$(mktemp -d)
hdiutil attach -nobrowse -noautoopen -readonly -mountpoint "$mnt" "$dl/$dmg" </dev/null
trap 'hdiutil detach "$mnt" -quiet || true' EXIT
ls "$mnt"
rm -rf "$dest"
ditto "$mnt/Racket v$version" "$dest"

"$dest/bin/racket" -e '(printf "racket ~a, ~a\n" (version) (system-library-subpath #f))'
xattr -p com.apple.quarantine "$dest/bin/racket" 2>/dev/null || echo "no quarantine attribute"
codesign --display --verbose=2 "$dest/bin/racket" 2>&1 | grep -E '^(Executable|Authority|TeamIdentifier|CodeDirectory)' || true
codesign --display --entitlements - "$dest/bin/racket" 2>&1 || true
