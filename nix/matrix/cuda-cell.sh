skip() {
  echo "matrix-skip: $1"
  exit 77
}

min_free=${MATRIX_MIN_FREE_GB:-30}
free_gb=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo)
load=$(cut -d' ' -f1 /proc/loadavg)
max_load=${MATRIX_MAX_LOAD:-$(nproc)}
[ "$free_gb" -ge "$min_free" ] || skip "${free_gb} GB available < ${min_free}"
awk -v l="$load" -v c="$max_load" 'BEGIN { exit !(l < c) }' \
  || skip "load ${load} >= ${max_load} cores"

set -m
work=$(mktemp -d -t torch-matrix-cuda.XXXXXX)
cleanup() {
  local g
  for g in $(jobs -p); do drain "$g"; done
  chmod -R u+w "$work"
  rm -rf "$work"
}
trap cleanup EXIT
trap 'trap "" INT TERM; exit 130' INT
trap 'trap "" INT TERM; exit 143' TERM

step() {
  "$@" </dev/null &
  wait "$!"
}

bash "$CELL_DRIVER_FARM" "$work/driver" || skip "no NVIDIA driver on this host"

# Replaced, not prepended: the host's own CUDA and cuDNN would shadow libtorch's.
export LD_LIBRARY_PATH="$work/driver:$CELL_CUDNN"
export RKTORCH_CUDA_DRIVER_PATH="$work/driver"
export HOME="$work/home"
mkdir -p "$HOME"
export PLTUSERHOME="$work/plt"

echo "matrix-stage: gtests"
step "$CELL_SHIM/libexec/torchrkt/torchrkt_tests"

cp -r "$CELL_SRC" "$work/src"
chmod -R u+w "$work/src"
cd "$work/src" || exit
step bash "$CELL_RACKET_CELL" install "$CELL_SHIM" "$CELL_DEPS" docs
step bash "$CELL_RACKET_CELL" test
