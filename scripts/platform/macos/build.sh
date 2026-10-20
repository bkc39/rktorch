#!/usr/bin/env bash
#
#   scripts/platform/macos/build.sh <work-dir> [codecs|gtest|shim|bundle|audit|gtests]...
set -euo pipefail

usage="usage: build.sh <work-dir> [codecs|gtest|shim|bundle|audit|gtests]..."
mkdir -p "${1:?$usage}"
work=$(cd "$1" && pwd)
shift
steps=("$@")
[ "${#steps[@]}" -gt 0 ] || steps=(codecs gtest shim bundle audit gtests)
here=$(cd "$(dirname "$0")" && pwd)
jobs=${JOBS:-$(sysctl -n hw.ncpu)}
target=$(cat "$here/deployment-target")
export MACOSX_DEPLOYMENT_TARGET=$target
libtorch_zip="libtorch-macos-arm64-2.14.0.zip"
availability="-Werror=unguarded-availability -Werror=unguarded-availability-new"
codec_cflags="-O2 -fPIC -mmacosx-version-min=$target $availability"
codec_ldflags="-Wl,-headerpad_max_install_names"
toolchain="$(xcrun clang --version | sed -n 1p), SDK $(xcrun --show-sdk-version)"
# shellcheck source=scripts/platform/build-common.sh
. "$here/../build-common.sh"
configure_flags["mpg123-1.33.4.tar.bz2"]+=" --with-cpu=aarch64"

commands() {
  "$here/load-commands.sh" "$1"
}

collect() {
  local queue lib cmd path name loads
  rm -rf "$bundle"
  mkdir -p "$bundle"
  cp "$work/shim-build/libtorchrkt.dylib" "$bundle/"
  strip -x "$bundle/libtorchrkt.dylib"
  queue=("$bundle/libtorchrkt.dylib")
  while [ "${#queue[@]}" -gt 0 ]; do
    lib=${queue[0]}
    queue=("${queue[@]:1}")
    loads=$(commands "$lib")
    while read -r cmd path; do
      case $cmd:$path in
        LC_ID_DYLIB:* | LC_RPATH:* | LC_BUILD_VERSION:* | LC_VERSION_MIN_MACOSX:*) continue ;;
        *:/usr/lib/* | *:/System/*) continue ;;
      esac
      name=${path##*/}
      [ ! -e "$bundle/$name" ] || continue
      if [ -f "$libtorch/lib/$name" ]; then
        cp "$libtorch/lib/$name" "$bundle/"
      elif [ -e "$deps/lib/$name" ]; then
        cp -L "$deps/lib/$name" "$bundle/"
        strip -x "$bundle/$name"
      else
        continue
      fi
      queue+=("$bundle/$name")
    done <<<"$loads"
  done
  chmod 0644 "$bundle"/*
}

relink() {
  local lib name args rpath cmd path loads
  for lib in "$bundle"/*.dylib; do
    name=${lib##*/}
    args=(-id "@rpath/$name")
    rpath=no
    loads=$(commands "$lib")
    while read -r cmd path; do
      case $cmd:$path in
        LC_RPATH:@loader_path | LC_RPATH:@loader_path/) rpath=yes ;;
        LC_RPATH:*) args+=(-delete_rpath "$path") ;;
        LC_ID_DYLIB:* | LC_BUILD_VERSION:* | LC_VERSION_MIN_MACOSX:*) ;;
        *:/usr/lib/* | *:/System/* | *:@rpath/* | *:@loader_path/*) ;;
        *) args+=(-change "$path" "@rpath/${path##*/}") ;;
      esac
    done <<<"$loads"
    [ "$rpath" = yes ] || args+=(-add_rpath @loader_path)
    install_name_tool "${args[@]}" "$lib" 2>&1 | { grep -v 'will invalidate the code signature' || true; }
    codesign --force --sign - "$lib"
  done
}

fetch_sources
for step_name in "${steps[@]}"; do
  case $step_name in
    codecs) build_codecs ;;
    gtest) build_gtest "-DCMAKE_OSX_DEPLOYMENT_TARGET=$target" ;;
    shim)
      unpack_libtorch
      stage_stb
      build_shim "-DCMAKE_OSX_DEPLOYMENT_TARGET=$target" \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        "-DCMAKE_C_FLAGS=$availability" \
        "-DCMAKE_CXX_FLAGS=$availability" \
        -DCMAKE_INSTALL_RPATH=@loader_path \
        -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-headerpad_max_install_names
      ;;
    bundle)
      step "bundle"
      collect
      relink
      ls -l "$bundle"
      report_bundle
      ;;
    audit)
      step "audit, deployment target $target"
      "$here/audit-bundle.sh" "$bundle" "$target" | tee "$log/audit.txt"
      ;;
    gtests) run_gtests DYLD_LIBRARY_PATH ;;
    *)
      echo "$usage" >&2
      exit 2
      ;;
  esac
done
