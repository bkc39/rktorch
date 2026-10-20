#!/usr/bin/env bash
#
#   scripts/platform/macos/load-commands.sh <dylib>
set -euo pipefail

otool -l "${1:?usage: load-commands.sh <dylib>}" | awk '
  $1 == "cmd" { cmd = $2; next }
  $1 == "name" && cmd ~ /^LC_(ID|LOAD|LOAD_WEAK|REEXPORT|LOAD_UPWARD|LAZY_LOAD)_DYLIB$/ { print cmd, $2; next }
  $1 == "path" && cmd == "LC_RPATH" { print cmd, $2; next }
  $1 == "minos" && cmd == "LC_BUILD_VERSION" { print cmd, $2; next }
  $1 == "version" && cmd == "LC_VERSION_MIN_MACOSX" { print cmd, $2; next }'
