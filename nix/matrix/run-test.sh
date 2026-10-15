#!/usr/bin/env bash
set -euo pipefail

runner=$1/run.sh
cell=$1/cuda-cell.sh
shift
t=$(mktemp -d)
bash=$(command -v bash)
set -m
stubs=()
cleanup() {
  local pid
  for pid in "${stubs[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

mkdir -p "$t/bin" "$t/fixtures" "$t/out/good" "$t/out/skiprun/bin" "$t/out/failrun/bin" \
  "$t/out/rackfail/bin" "$t/out/gtestrun/bin"

cat >"$t/bin/nix" <<EOF
#!$bash
for arg; do drv=\${arg%^out}; done
name=\${drv##*/}
case "\$name" in
  bad) echo "error: builder for '\$drv' failed with exit code 1" >&2; exit 1 ;;
  hang) echo \$\$ >"$t/hang.pid"; exec sleep 300 ;;
  deaf) trap '' TERM; echo \$\$ >"$t/hang.pid"; exec sleep 300 ;;
  slow) sleep 1.5 ;;
esac
if [ -f "$t/fixtures/\$name.log" ]; then
  cat "$t/fixtures/\$name.log" >&2
  exit 1
fi
echo "$t/out/\$name"
EOF

cat >"$t/bin/vm_stat" <<'EOF'
#!/bin/sh
echo "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
echo "Pages free:                               100000."
echo "Pages active:                             900000."
echo "Pages inactive:                           150000."
echo "Pages speculative:                         12144."
EOF

cat >"$t/out/skiprun/bin/cell" <<EOF
#!$bash
echo "matrix-skip: no NVIDIA driver on this host"
exit 77
EOF
cat >"$t/out/failrun/bin/cell" <<EOF
#!$bash
echo "matrix-stage: gtests"
echo "[  FAILED  ] Ops.Add"
exit 1
EOF
cat >"$t/out/rackfail/bin/cell" <<EOF
#!$bash
echo "matrix-stage: test"
echo 'raco test: (file "./torch/foreign/error.rkt")'
echo "--------------------"
echo "modes stay with their thread"
echo "ERROR"
echo ""
echo "result arity mismatch;"
exit 1
EOF
cat >"$t/out/gtestrun/bin/cell" <<EOF
#!$bash
echo "matrix-stage: gtests"
cat <<'LOG'
[==========] Running 212 tests from 31 test suites.
[ RUN      ] TorchrktAutograd.GradBeforeBackwardErrors
[       OK ] TorchrktAutograd.GradBeforeBackwardErrors (0 ms)
[ RUN      ] TorchrktOps.Add
/build/cpp/tests/ops_test.cpp:31: Failure
Expected equality of these values:
[  FAILED  ] TorchrktOps.Add (1 ms)
[==========] 212 tests from 31 test suites ran. (840 ms total)
[  FAILED  ] 1 test, listed below:
[  FAILED  ] TorchrktOps.Add
LOG
exit 1
EOF
chmod +x "$t/bin/nix" "$t/bin/vm_stat" "$t"/out/*/bin/cell

cat >"$t/fixtures/shimfail.log" <<'EOF'
torchrkt-cpp> -- Automatic GPU detection failed. Building for common architectures.
torchrkt-cpp> [2/56] Building CXX object CMakeFiles/torchrkt.dir/src/torchrkt/detail/error.cpp.o
torchrkt-cpp> FAILED: CMakeFiles/torchrkt.dir/src/torchrkt/ops.cpp.o
torchrkt-cpp> /build/cpp/src/torchrkt/ops.cpp:42:7: error: use of undeclared identifier 'at_add'
torchrkt-cpp> ninja: build stopped: subcommand failed.
error: builder for '/nix/store/aaaa-torchrkt-cpp-0.1.0.drv' failed with exit code 1
EOF
cat >"$t/fixtures/toolfail.log" <<'EOF'
racket> gcc -pthread -Ics/c/rktio/ -I../rktio/ -g -O2 -Wall -o cs/c/rktio/rktio_error.o -c ../rktio/rktio_error.c
racket> ar rc cs/c/rktio/librktio.a cs/c/rktio/rktio_fs.o cs/c/rktio/rktio_error.o
racket> make[1]: *** [Makefile:93: cs/c/racketcs] Killed
error: Cannot build '/nix/store/bbbb-racket-9.1.drv'.
       Reason: builder failed with exit code 2.
       Last 25 log lines:
       > gcc -o cs/c/rktio/rktio_error.o -c ../rktio/rktio_error.c
EOF

printf '%s\t%s\t%s\t%s\t%s\n' \
  9.1 zip nix build "shim=/s/good toolchain=/s/good cell=/s/good" \
  9.2 zip nix build "shim=/s/good cell=/s/bad" \
  9.2 cached nix skip "2.12.0 < floor 2.14" \
  9.3 cuda nix build "shim=/s/good run=/s/skiprun" \
  9.3 gfail nix build "run=/s/failrun" \
  9.3 rack nix build "run=/s/rackfail" \
  9.3 gtest nix build "shim=/s/good run=/s/gtestrun" \
  9.3 shimlog nix build "shim=/s/shimfail" \
  9.3 toollog nix build "shim=/s/good toolchain=/s/toolfail" \
  9.3 hang nix build "shim=/s/hang" \
  9.3 deaf nix build "shim=/s/deaf" \
  9.3 slow nix build "shim=/s/slow toolchain=/s/slow" >"$t/cells"

row() { echo "^aarch64-darwin +$1 +nix +[0-9]+s +$2\$"; }

run() {
  PATH="$t/bin:$PATH" MATRIX_SYSTEM="${MATRIX_SYSTEM:-aarch64-darwin}" MATRIX_CELLS="$t/cells" \
    MATRIX_LOG_DIR="$t/logs" MATRIX_MIN_FREE_GB="${MATRIX_MIN_FREE_GB:-0}" \
    timeout -k 5 60 bash -euo pipefail "$runner" "$@" 2>"$t/err"
}

expect() {
  grep -E "$1" "$t/table" >/dev/null || {
    echo "run-test: $case: no row matches: $1" >&2
    cat "$t/table" "$t/err" >&2
    exit 1
  }
}

status_is() {
  [ "$status" = "$1" ] || {
    echo "run-test: $case: exit $status, expected $1" >&2
    cat "$t/table" "$t/err" >&2
    exit 1
  }
}

case_rows() {
  status=0
  run 'zip' 'cached' 'cuda' 'gfail' 'rack' >"$t/table" || status=$?
  status_is 1
  expect "$(row '9\.1 +zip' 'pass')"
  expect "$(row '9\.2 +zip' "fail \(cell\): error: builder for .*/s/bad. failed with exit code 1")"
  expect "$(row '9\.2 +cached' 'skip: 2\.12\.0 < floor 2\.14')"
  expect "$(row '9\.3 +cuda' 'skip: no NVIDIA driver on this host')"
  expect "$(row '9\.3 +gfail' 'fail \(gtests\): \[  FAILED  \] Ops\.Add')"
  expect "$(row '9\.3 +rack' 'fail \(test\): ERROR modes stay with their thread: result arity mismatch;')"
  [ "$(grep -c '^aarch64-darwin' "$t/table")" = 6 ]
  status=0
  run '9.2 cached' >"$t/table" || status=$?
  status_is 0
  [ "$(grep -c '^aarch64-darwin' "$t/table")" = 1 ]
}

case_no_match() {
  status=0
  run '9.0' 'nonesuch' >"$t/table" || status=$?
  status_is 2
  grep -qx "torch-matrix: no cell matches '9.0' 'nonesuch'" "$t/err" || {
    echo "run-test: $case: no 'no cell matches' error" >&2
    cat "$t/table" "$t/err" >&2
    exit 1
  }
}

case_failure_line() {
  status=0
  run 'gtest' 'shimlog' 'toollog' >"$t/table" || status=$?
  status_is 1
  expect "$(row '9\.3 +gtest' 'fail \(gtests\): \[  FAILED  \] TorchrktOps\.Add \(1 ms\)')"
  expect "$(row '9\.3 +shimlog' "fail \(shim\): /build/cpp/src/torchrkt/ops\.cpp:42:7: error: use of undeclared identifier 'at_add'")"
  expect "$(row '9\.3 +toollog' "fail \(toolchain\): error: Cannot build '/nix/store/bbbb-racket-9\.1\.drv'\.")"
}

state() {
  local stat
  if [ -r /proc/self/stat ]; then
    { read -r stat <"/proc/$1/stat"; } 2>/dev/null || return 1
    stat=${stat##*") "}
  else
    stat=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
    stat=${stat#"${stat%%[! ]*}"}
  fi
  echo "${stat:0:1}"
}

alive() {
  local s
  if [ -r /proc/self/stat ] || command -v ps >/dev/null; then
    s=$(state "$1") && [ -n "$s" ] && [ "$s" != Z ]
  else
    kill -0 "$1" 2>/dev/null
  fi
}

await_hang() {
  local tries=0
  until [ -s "$t/hang.pid" ]; do
    tries=$((tries + 1))
    [ "$tries" -lt 100 ] || { echo "run-test: $case: the stub never started" >&2; exit 1; }
    sleep 0.1
  done
  hang=$(cat "$t/hang.pid")
  stubs+=("$hang")
}

hang_gone() {
  local started=$1 hung=$2
  [ $((SECONDS - started)) -le 10 ] || { echo "run-test: $case: took $((SECONDS - started))s" >&2; exit 1; }
  if alive "$hang"; then
    echo "run-test: $case: the running step $hang outlived the runner" >&2
    exit 1
  fi
  expect "$(row '9\.1 +zip' 'pass')"
  expect "$(row "9\\.3 +$hung" 'fail \(shim\): interrupted')"
  grep -q '^logs: ' "$t/table"
}

interrupt() {
  rm -f "$t/hang.pid"
  PATH="$t/bin:$PATH" MATRIX_SYSTEM=aarch64-darwin MATRIX_CELLS="$t/cells" \
    MATRIX_LOG_DIR="$t/logs" MATRIX_MIN_FREE_GB=0 \
    timeout -k 20 60 bash -euo pipefail "$runner" '9.1 zip' "$1" >"$t/table" 2>"$t/err" &
  local pid=$! started
  await_hang
  started=$SECONDS
  kill -INT -- -"$pid"
  status=0
  wait "$pid" || status=$?
  status_is 130
  hang_gone "$started" "$1"
}

case_interrupt() { interrupt hang; }
case_interrupt_deaf() { interrupt deaf; }

case_terminate() {
  rm -f "$t/hang.pid"
  local started=$SECONDS
  status=0
  PATH="$t/bin:$PATH" MATRIX_SYSTEM=aarch64-darwin MATRIX_CELLS="$t/cells" \
    MATRIX_LOG_DIR="$t/logs" MATRIX_MIN_FREE_GB=0 \
    timeout -k 5 60 timeout --preserve-status 4 bash -euo pipefail "$runner" '9.1 zip' 'hang' \
    >"$t/table" 2>"$t/err" &
  local pid=$!
  await_hang
  wait "$pid" || status=$?
  status_is 143
  hang_gone "$started" hang
}

case_deadline() {
  local started=$SECONDS
  status=0
  MATRIX_CELL_TIMEOUT=2s run 'slow' >"$t/table" || status=$?
  status_is 1
  expect "$(row '9\.3 +slow' 'fail \(toolchain\): timed out after 2s')"
  [ $((SECONDS - started)) -le 5 ] || { echo "run-test: $case: took $((SECONDS - started))s" >&2; exit 1; }
}

case_guard() {
  status=0
  MATRIX_MIN_FREE_GB=5 run '9.1 zip' 'cached' >"$t/table" || status=$?
  status_is 2
  expect "$(row '9\.1 +zip' 'skip: 4 GB available < 5')"
  status=0
  MATRIX_MIN_FREE_GB=4 run '9.1 zip' 'cached' >"$t/table" || status=$?
  status_is 0
  expect "$(row '9\.1 +zip' 'pass')"
  status=0
  run 'cuda' >"$t/table" || status=$?
  status_is 2
  if [ -r /proc/meminfo ]; then
    status=0
    MATRIX_SYSTEM=x86_64-linux MATRIX_MIN_FREE_GB=100000 run '9.1 zip' >"$t/table" || status=$?
    status_is 2
    grep -Eq 'skip: [0-9]+ GB available < 100000$' "$t/table" || { echo "run-test: $case: no linux guard row" >&2; exit 1; }
  fi
  status=0
  MATRIX_SYSTEM=test-system MATRIX_MIN_FREE_GB=100000 run '9.1 zip' >"$t/table" || status=$?
  status_is 0
  grep -Eq 'pass \(free memory unread, guard off\)$' "$t/table" || { echo "run-test: $case: no unread note" >&2; cat "$t/table" >&2; exit 1; }
}

cuda_cell() {
  [ -r /proc/loadavg ] || return 0
  rm -rf "$t/cuda"
  mkdir -p "$t/cuda/shim/libexec/torchrkt" "$t/cuda/tmp"
  cat >"$t/cuda/shim/libexec/torchrkt/torchrkt_tests" <<EOF
#!$bash
$1
echo \$\$ >"$t/cuda/gtests.pid"
exec sleep 300
EOF
  cat >"$t/cuda/farm" <<'EOF'
mkdir -p "$1"
EOF
  chmod +x "$t/cuda/shim/libexec/torchrkt/torchrkt_tests"
  local started pid
  TMPDIR="$t/cuda/tmp" MATRIX_MIN_FREE_GB=0 CELL_SRC=/nonexistent CELL_SHIM="$t/cuda/shim" \
    CELL_DEPS=/nonexistent CELL_CUDNN=/nonexistent CELL_RACKET_CELL=/nonexistent \
    CELL_DRIVER_FARM="$t/cuda/farm" \
    timeout -k 20 60 bash -euo pipefail "$cell" >"$t/table" 2>"$t/err" &
  pid=$!
  until [ -s "$t/cuda/gtests.pid" ] || ! alive "$pid"; do sleep 0.1; done
  if grep -q '^matrix-skip: load' "$t/table"; then
    wait "$pid" || true
    echo "run-test: $case: skipped, the host is loaded" >&2
    return 0
  fi
  [ -s "$t/cuda/gtests.pid" ] || {
    echo "run-test: $case: gtests never started" >&2
    cat "$t/table" "$t/err" >&2
    exit 1
  }
  hang=$(cat "$t/cuda/gtests.pid")
  stubs+=("$hang")
  started=$SECONDS
  kill -TERM -- -"$pid"
  status=0
  wait "$pid" || status=$?
  status_is 143
  [ $((SECONDS - started)) -le 10 ] || { echo "run-test: $case: took $((SECONDS - started))s" >&2; exit 1; }
  if alive "$hang"; then
    echo "run-test: $case: gtests $hang outlived the cell" >&2
    exit 1
  fi
  [ -z "$(ls -A "$t/cuda/tmp")" ] || { echo "run-test: $case: the work dir survived" >&2; exit 1; }
}

case_cuda_cell() { cuda_cell ''; }
case_cuda_cell_deaf() { cuda_cell "trap '' TERM"; }

case_zombie() {
  [ -r /proc/self/stat ] || return 0
  rm -f "$t/zombie.pid"
  (
    parent=$BASHPID
    (
      n=0
      until grep -qx sleep "/proc/$parent/comm" || [ "$n" -ge 200 ]; do n=$((n + 1)); sleep 0.05; done
    ) &
    echo $! >"$t/zombie.pid"
    exec sleep 30
  ) &
  local sleeper=$! zombie tries=0
  stubs+=("$sleeper")
  until [ -s "$t/zombie.pid" ] && zombie=$(cat "$t/zombie.pid") && [ "$(state "$zombie")" = Z ]; do
    tries=$((tries + 1))
    [ "$tries" -lt 200 ] || { echo "run-test: $case: no zombie appeared" >&2; exit 1; }
    sleep 0.05
  done
  kill -0 "$zombie"
  if alive "$zombie"; then
    echo "run-test: $case: the zombie $zombie counts as alive" >&2
    exit 1
  fi
  alive "$sleeper" || { echo "run-test: $case: the live sleeper $sleeper counts as dead" >&2; exit 1; }
  kill -KILL "$sleeper"
  wait "$sleeper" 2>/dev/null || true
}

cases=("$@")
[ "${#cases[@]}" -gt 0 ] || cases=(rows no_match failure_line interrupt interrupt_deaf terminate deadline guard cuda_cell
    cuda_cell_deaf zombie)
for case in "${cases[@]}"; do
  "case_$case"
  echo "run-test: $case ok"
done
