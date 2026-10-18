#!/usr/bin/env bash
#
#   bash /src/scripts/platform/linux/in-container.sh   (run by build.sh)
set -euo pipefail

jobs=${JOBS:-4}
work=/work
toolchain=${IMAGE:?build.sh passes the image reference}
here=/src/scripts/platform/linux
libtorch_zip="libtorch-shared-with-deps-2.14.0-cpu.zip"
codec_cflags="-O2 -fPIC"
codec_ldflags=""
# shellcheck source=scripts/platform/build-common.sh
. /src/scripts/platform/build-common.sh
configure_flags["mpg123-1.33.4.tar.bz2"]+=" --with-cpu=x86-64"

fetch_sources
unpack_libtorch
stage_stb
build_codecs
build_gtest

# --exclude-libs keeps gcc-toolset's static libstdc++_nonshared pieces out of
# the shim's exports.
LD_LIBRARY_PATH=$deps/lib:$libtorch/lib build_shim \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
  "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--no-undefined -Wl,--exclude-libs,ALL -Wl,--enable-new-dtags -Wl,--as-needed"

rm -rf "$bundle"
mkdir -p "$bundle"
strip --strip-unneeded -o "$bundle/libtorchrkt.so" "$work/shim-build/libtorchrkt.so"
queue=("$bundle/libtorchrkt.so")
while [ "${#queue[@]}" -gt 0 ]; do
  lib=${queue[0]}
  queue=("${queue[@]:1}")
  for needed in $(readelf -d "$lib" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
    [ ! -e "$bundle/$needed" ] || continue
    if [ -f "$libtorch/lib/$needed" ]; then
      cp "$libtorch/lib/$needed" "$bundle/"
    elif [ -e "$deps/lib/$needed" ]; then
      strip --strip-unneeded -o "$bundle/$needed" "$deps/lib/$needed"
      patchelf --set-rpath '$ORIGIN' "$bundle/$needed"
    else
      continue
    fi
    queue+=("$bundle/$needed")
  done
done
chmod 0644 "$bundle"/*

bash "$here/audit-bundle.sh" "$bundle" | tee "$log/audit.txt"
run_gtests LD_LIBRARY_PATH
report_bundle
