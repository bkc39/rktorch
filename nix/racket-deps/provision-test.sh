#!/usr/bin/env bash
set -euo pipefail

provision=$1
f=$2
t=$(mktemp -d)
trap 'chmod -R u+w "$t"; rm -rf "$t"' EXIT
app=$t/rdfix-app
keep=$t/keep
log=$t/log
mkdir -p "$keep/rdfix-k"

rd() { racket "$(dirname "$provision")/racket-deps.rkt" "$@"; }
mkapp() {
  rm -rf "$app"
  mkdir -p "$app"
  cp "$f/app/main.rkt" "$f/$1/info.rkt" "$app/"
  chmod -R u+w "$app"
}
mkout() {
  local out=$t/$1
  shift
  mkdir -p "$out"
  for p in "$@"; do cp -r "$f/$p" "$out/"; done
  chmod -R u+w "$out"
  rd manifest "$app" > "$out/.info-deps"
}
provision() { bash "$provision" "$t/$1" "$app" "$keep" > "$log" 2>&1; }
fails() {
  if "$@"; then
    echo "expected a failure: $*" >&2
    exit 1
  fi
}
installed() { rd installed | sort | paste -sd' '; }
stamps() {
  for s in "$PLTUSERHOME"/.deps*; do
    if [ -e "$s" ]; then basename "$s"; fi
  done | paste -sd' '
}
expect() {
  [ "$1" = "$2" ] || {
    echo "expected '$2', got '$1'" >&2
    cat "$log" >&2
    exit 1
  }
}

export PLTUSERHOME=$t/plt
mkapp app-ab
mkout out-1 rdfix-a rdfix-b rdfix-k
provision out-1
expect "$(installed)" "rdfix-a rdfix-app rdfix-b rdfix-k"
first=$(stamps)
expect "$(wc -w <<< "$first")" 1
provision out-1
[ ! -s "$log" ] || { echo "second provision was not a no-op" >&2; cat "$log" >&2; exit 1; }

mkapp app-a
mkout out-2 rdfix-a
provision out-2
expect "$(installed)" "rdfix-a rdfix-app rdfix-k"
second=$(stamps)
expect "$(wc -w <<< "$second")" 1
[ "$second" != "$first" ]

mkapp app-drift
fails provision out-2
grep -q 'bump racket-deps outputHash' "$log"
expect "$(stamps)" "$second"

export PLTUSERHOME=$t/plt-fresh
mkout out-3 rdfix-a
fails provision out-3
grep -q 'missing dependencies' "$log"
grep -q '^ *rdfix-c$' "$log"
expect "$(stamps)" ""
