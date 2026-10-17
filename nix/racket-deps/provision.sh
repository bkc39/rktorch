#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

[ $# -eq 4 ] || {
  echo "usage: provision.sh <racket-deps> <pkg-dir> <keep-tree> <platform-pkgs-dir>" >&2
  exit 2
}
deps=$1
pkg_dir=$2
keep=$3
platform_dir=$4

rd() { racket "$(dirname "${BASH_SOURCE[0]}")/racket-deps.rkt" "$@"; }
names() { for d in "$@"; do if [ -d "$d" ]; then basename "$d"; fi; done | sort -u; }

pkg=$(basename "$pkg_dir")
ver=$(racket --version 2>&1 | grep -oE 'v[0-9]+\.[0-9]+' | tr -d v | tr . -)
info_sha=$(sha256sum "$pkg_dir/info.rkt" | cut -c1-16)
stamp="$PLTUSERHOME/.deps5-$(basename "$deps")-$ver-$info_sha"
record="$PLTUSERHOME/.racket-deps-pkgs"
[ ! -f "$stamp" ] || exit 0

echo "Installing Racket package $pkg (link mode, Racket $ver)..."
rd check "$pkg_dir" "$deps"
mkdir -p "$PLTUSERHOME"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
names "$deps"/*/ > "$tmp/new"
{ names "$keep"/*/; echo "$pkg"; cat "$tmp/new"; } | sort -u > "$tmp/held"
rd installed | sort -u > "$tmp/installed"
{ [ ! -f "$record" ] || cat "$record"; } | sort -u > "$tmp/old"
comm -12 "$tmp/old" "$tmp/installed" | comm -23 - "$tmp/held" > "$tmp/stale"
mapfile -t stale < "$tmp/stale"
if [ "${#stale[@]}" -gt 0 ]; then
  echo "Removing packages the previous racket-deps held: ${stale[*]}"
  raco pkg remove --batch --no-setup --force --scope user "${stale[@]}"
fi
raco pkg install --batch --copy --no-docs --no-setup --scope user \
  --skip-installed "$deps"/*/
raco pkg update --batch --copy --no-docs --no-setup --no-trash --deps fail \
  --scope user "$deps"/*/
rd install-platform "$pkg_dir" "$platform_dir" --link
raco pkg install --batch --deps fail --no-setup --link --scope user \
  --skip-installed --name "$pkg" "$pkg_dir"
raco setup --no-docs --pkgs "$pkg"
cp "$tmp/new" "$record"
rm -f "$PLTUSERHOME"/.deps[0-9]*
touch "$stamp"
