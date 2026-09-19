#!/usr/bin/env bash
# Fail when a tracked file carries an email address (#169).
#
# The package catalog installs straight from git and copies tracked metadata,
# so an address in the tree is published and scraped. Publish a handle instead.
# Third-party notices and licence text keep whatever their authors wrote.
#
# Run: scripts/check-no-addresses.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

address='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}'

# A module reference such as `foo@bar.rkt` has an address's shape, so a token
# whose last component is a source extension is not one. `.sh` is a real
# top-level domain and a script suffix; the suffix is the likelier reading in
# this tree.
extensions='rkt|scrbl|cpp|hpp|cc|[ch]|py|md|txt|sh|nix|ya?ml|json|lock|so|dylib|toml|cfg|patch'

allowed=(
  ':!scripts/check-no-addresses.sh'
  ':!*NOTICE*'
  ':!*LICENSE*'
  ':!*COPYING*'
  ':!*third-party*'
)

tokens=$(git grep -hIoE "$address" -- "${allowed[@]}" \
         | sort -u \
         | grep -vEi "\.($extensions)\$" \
         || true)

if [ -z "$tokens" ]; then
  echo "no addresses in tracked files"
  exit 0
fi

echo "tracked files carry email addresses; publish a handle instead (#169):" >&2
while IFS= read -r token; do
  git grep -nF -- "$token" | sed 's/^/  /' >&2
done <<< "$tokens"
exit 1
