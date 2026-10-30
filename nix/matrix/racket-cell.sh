#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

stage() { echo "matrix-stage: $1"; }

case "$1" in
  install)
    shim=$2
    deps=$3
    docs=$4
    jobs=${NIX_BUILD_CORES:-4}
    export TORCHRKT_NATIVE_LIB_PATH=$shim
    mkdir -p "$PLTUSERHOME" torch/native-libs
    stage install
    raco pkg install --batch --copy --no-docs --no-setup --scope user "$deps"/*/
    # libtorch is reached through the shim's rpath, so only the shim is staged.
    install -m 0555 "$shim"/lib/libtorchrkt.* torch/native-libs/
    racket nix/racket-deps/racket-deps.rkt install-platform ./torch ./pkgs --copy
    raco pkg install --batch --deps fail --no-setup --copy --scope user \
      --name torch ./torch
    stage "setup ($docs)"
    case "$docs" in
      docs) raco setup -j "$jobs" --pkgs torch ;;
      no-docs) raco setup -j "$jobs" --no-docs --pkgs torch ;;
      *) echo "racket-cell: docs or no-docs, got $docs" >&2; exit 2 ;;
    esac
    ;;
  test)
    stage test
    raco test --timeout 300 ./torch/
    raco test examples/test/
    raco test --timeout 1200 scripts/bench/
    ;;
  *)
    echo "usage: racket-cell.sh install <shim> <racket-deps> docs|no-docs | test" >&2
    exit 2
    ;;
esac
