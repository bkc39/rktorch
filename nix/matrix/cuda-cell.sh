# The CUDA cell runs on the host, since the build sandbox has no driver.
# CELL_SRC CELL_SHIM CELL_DEPS CELL_CUDNN CELL_RACKET_CELL CELL_DRIVER_FARM
# come from the flake; exit 77 is a skip.

skip() {
  echo "matrix-skip: $1"
  exit 77
}

min_free=${MATRIX_MIN_FREE_GB:-30}
step_timeout=${MATRIX_STEP_TIMEOUT:-2h}
free_gb=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo)
load=$(cut -d' ' -f1 /proc/loadavg)
cores=$(nproc)
[ "$free_gb" -ge "$min_free" ] || skip "${free_gb} GB available < ${min_free}"
awk -v l="$load" -v c="$cores" 'BEGIN { exit !(l < c) }' \
  || skip "load ${load} >= ${cores} cores"

work=$(mktemp -d -t torch-matrix-cuda.XXXXXX)
trap 'chmod -R u+w "$work"; rm -rf "$work"' EXIT
bash "$CELL_DRIVER_FARM" "$work/driver" || skip "no NVIDIA driver on this host"

# Replaced, not prepended: the host's own CUDA and cuDNN would shadow libtorch's.
export LD_LIBRARY_PATH="$work/driver:$CELL_CUDNN"
export RKTORCH_CUDA_DRIVER_PATH="$work/driver"
export HOME="$work/home"
mkdir -p "$HOME"
export PLTUSERHOME="$work/plt"

echo "matrix-stage: gtests"
timeout "$step_timeout" "$CELL_SHIM/libexec/torchrkt/torchrkt_tests"

cp -r "$CELL_SRC" "$work/src"
chmod -R u+w "$work/src"
cd "$work/src" || exit
timeout "$step_timeout" bash "$CELL_RACKET_CELL" install "$CELL_SHIM" "$CELL_DEPS" docs
timeout "$step_timeout" bash "$CELL_RACKET_CELL" test
