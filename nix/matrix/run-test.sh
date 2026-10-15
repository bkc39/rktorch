#!/usr/bin/env bash
# Drives run.sh against a stub `nix` over cells that pass, fail, skip and run.
#   run-test.sh <run.sh>
set -euo pipefail

runner=$1
t=$(mktemp -d)
bash=$(command -v bash)
mkdir -p "$t/bin" "$t/out/skiprun/bin" "$t/out/failrun/bin" "$t/out/good"

cat >"$t/bin/nix" <<EOF
#!$bash
for arg; do drv=\${arg%^out}; done
case "\$drv" in
  */bad) echo "error: builder for '\$drv' failed with exit code 1" >&2; exit 1 ;;
  *) echo "$t/out/\${drv##*/}" ;;
esac
EOF
cat >"$t/out/skiprun/bin/cell" <<EOF
#!$bash
cat >/dev/null
echo "matrix-skip: no NVIDIA driver on this host"
exit 77
EOF
cat >"$t/out/failrun/bin/cell" <<EOF
#!$bash
echo "matrix-stage: gtests"
echo "[  FAILED  ] Ops.Add"
exit 1
EOF
chmod +x "$t/bin/nix" "$t"/out/*/bin/cell

printf '%s\t%s\t%s\t%s\t%s\n' \
  9.0 zip nix build "shim=/s/good toolchain=/s/good cell=/s/good" \
  9.1 zip nix build "shim=/s/good cell=/s/bad" \
  9.2 cached nix skip "2.12.0 < floor 2.14" \
  9.3 cuda nix build "shim=/s/good run=/s/skiprun" \
  9.3 cuda2 nix build "run=/s/failrun" >"$t/cells"

run() {
  PATH="$t/bin:$PATH" MATRIX_SYSTEM=test-system MATRIX_CELLS="$t/cells" \
    MATRIX_LOG_DIR="$t/logs" MATRIX_MIN_FREE_GB="${MATRIX_MIN_FREE_GB:-0}" \
    bash -euo pipefail "$runner" "$@" 2>/dev/null
}

expect() {
  grep -E "$1" "$t/table" >/dev/null || {
    echo "run-test: no row matches: $1" >&2
    cat "$t/table" >&2
    exit 1
  }
}

status=0
run >"$t/table" || status=$?
[ "$status" = 1 ] || { echo "run-test: a failing cell must exit 1, got $status" >&2; exit 1; }
expect '^test-system +9\.0 +zip +nix +[0-9]+s +pass$'
expect '^test-system +9\.1 +zip +nix +[0-9]+s +fail \(cell\): error: builder for .*/s/bad. failed'
expect '^test-system +9\.2 +cached +nix +[0-9]+s +skip: 2\.12\.0 < floor 2\.14$'
expect '^test-system +9\.3 +cuda +nix +[0-9]+s +skip: no NVIDIA driver on this host$'
expect '^test-system +9\.3 +cuda2 +nix +[0-9]+s +fail \(gtests\): \[  FAILED  \] Ops\.Add$'
[ "$(grep -c '^test-system' "$t/table")" = 5 ]

run '9.2 cached' >"$t/table"
expect '^test-system +9\.2 +cached'
[ "$(grep -c '^test-system' "$t/table")" = 1 ]

MATRIX_MIN_FREE_GB=100000 run '9.0' >"$t/table" || true
[ -r /proc/meminfo ] && expect '^test-system +9\.0 +zip +nix +[0-9]+s +skip: [0-9]+ GB available < 100000$'
echo "run-test: ok"
