#!/usr/bin/env bash
#
#   . scripts/platform/build-common.sh   (sourced by linux/in-container.sh and macos/build.sh,
#                                         which set work, jobs, libtorch_zip, codec_cflags
#                                         and codec_ldflags)
# shellcheck disable=SC2154

[ "${BASH_VERSINFO[0]}" -ge 4 ] || { echo "build: needs bash 4 or later, not $BASH_VERSION" >&2; exit 1; }

shared=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(cd "$shared/../.." && pwd)
dl=$work/dl
log=$work/log
deps=$work/deps
libtorch=$work/libtorch/libtorch
bundle=$work/bundle
gtest=googletest-1.17.0
mkdir -p "$dl" "$log"

codecs=(libogg-1.3.6.tar.gz libvorbis-1.3.7.tar.gz flac-1.5.0.tar.xz opus-1.5.2.tar.gz
        mpg123-1.33.4.tar.bz2 lame-3.100.tar.gz libsndfile-1.2.2.tar.xz)
declare -A configure_flags=(
  [flac-1.5.0.tar.xz]="--disable-programs --disable-examples --disable-doxygen-docs --disable-cpplibs"
  [opus-1.5.2.tar.gz]="--disable-doc --disable-extra-programs"
  [lame-3.100.tar.gz]="--disable-frontend --disable-decoder"
  [mpg123-1.33.4.tar.bz2]="--disable-components --enable-libmpg123"
  [libsndfile-1.2.2.tar.xz]="--disable-full-suite --disable-sqlite --disable-alsa"
)
declare -A pinned

step() {
  printf '\n== %s\n' "$*"
}

failed() {
  echo "build: $1 failed; see $2" >&2
  tail -30 "$2" >&2
  exit 1
}

current() {
  [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]
}

fetch_sources() {
  local fetched sum file
  fetched=$(bash "$shared/fetch.sh" "$dl" "$libtorch_zip" "${codecs[@]}" stb_image.h "$gtest.tar.gz")
  while read -r sum file; do
    pinned[$file]=$sum
  done <<<"$fetched"
}

unpack_libtorch() {
  if ! current "$work/libtorch/.pinned" "${pinned[$libtorch_zip]}"; then
    step "unpack $libtorch_zip"
    rm -rf "$work/libtorch"
    unzip -q -d "$work/libtorch" "$dl/$libtorch_zip"
    echo "${pinned[$libtorch_zip]}" >"$work/libtorch/.pinned"
  fi
}

stage_stb() {
  mkdir -p "$work/stb/include/stb" "$work/stb/lib/pkgconfig"
  cp -p "$dl/stb_image.h" "$work/stb/include/stb/"
  cat >"$work/stb/lib/pkgconfig/stb.pc" <<EOF
Name: stb
Description: stb single-file libraries
Version: 1
Cflags: -I$work/stb/include/stb
EOF
}

codecs_key() {
  local tarball
  echo "CFLAGS=$codec_cflags LDFLAGS=$codec_ldflags"
  for tarball in "${codecs[@]}"; do
    echo "${pinned[$tarball]} $tarball ${configure_flags[$tarball]-}"
  done
}

build_codecs() {
  local key tarball codec extra line sndfile_log
  key=$(codecs_key)
  current "$deps/.pinned" "$key" && return
  rm -rf "$deps" "$work/codec-build"
  mkdir -p "$work/codec-build"
  for tarball in "${codecs[@]}"; do
    codec=${tarball%.tar.*}
    step "$codec"
    tar -C "$work/codec-build" -xf "$dl/$tarball"
    if declare -F patch_codec >/dev/null; then
      patch_codec "$codec" "$work/codec-build/$codec"
    fi
    read -ra extra <<<"${configure_flags[$tarball]-}"
    (export PKG_CONFIG_LIBDIR=$deps/lib/pkgconfig CFLAGS=$codec_cflags CPPFLAGS="-I$deps/include" \
            LDFLAGS="-L$deps/lib $codec_ldflags"
     cd "$work/codec-build/$codec" &&
       ./configure --prefix="$deps" --disable-static --enable-shared "${extra[@]}" &&
       make -j"$jobs" && make install) >"$log/$codec.log" 2>&1 ||
      failed "$codec" "$log/$codec.log"
  done
  sndfile_log=$log/${codecs[-1]%.tar.*}.log
  for line in 'External FLAC/Ogg/Vorbis/Opus : [.]* yes' 'External MPEG Lame/MPG123 : [.]* yes'; do
    grep -q "$line" "$sndfile_log" || failed "libsndfile without its codecs" "$sndfile_log"
  done
  echo "$key" >"$deps/.pinned"
}

gtest_key() {
  echo "${pinned[$gtest.tar.gz]} $*"
}

build_gtest() {
  current "$work/gtest/.pinned" "$(gtest_key "$@")" && return
  step "$gtest"
  rm -rf "$work/gtest" "$work/gtest-build" "${work:?}/$gtest"
  tar -C "$work" -xf "$dl/$gtest.tar.gz"
  (cmake -S "$work/$gtest" -B "$work/gtest-build" -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_GMOCK=OFF -DCMAKE_INSTALL_PREFIX="$work/gtest" "$@" &&
     cmake --build "$work/gtest-build" -j "$jobs" &&
     cmake --install "$work/gtest-build") >"$log/googletest.log" 2>&1 ||
    failed googletest "$log/googletest.log"
  gtest_key "$@" >"$work/gtest/.pinned"
}

build_shim() {
  local key
  key=$(printf '%s\n' "${pinned[$libtorch_zip]}" "$(codecs_key)" "$(cat "$work/gtest/.pinned")")
  if ! current "$work/shim-build/.inputs" "$key"; then
    rm -rf "$work/shim-build"
    mkdir -p "$work/shim-build"
    echo "$key" >"$work/shim-build/.inputs"
  fi
  step "the shim"
  (export PKG_CONFIG_LIBDIR=$deps/lib/pkgconfig:$work/stb/lib/pkgconfig
   cmake -S "$src/cpp" -B "$work/shim-build" -G "Unix Makefiles" \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_TESTING=ON \
     "-DCMAKE_PREFIX_PATH=$libtorch;$work/gtest" \
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
     "$@" &&
     cmake --build "$work/shim-build" -j "$jobs") >"$log/shim.log" 2>&1 ||
    failed "the shim" "$log/shim.log"
}

run_gtests() {
  step "the gtests against the bundle"
  (export "$1=$bundle"
   exec "$work/shim-build/torchrkt_tests" --gtest_brief=1) >"$log/gtest.log" 2>&1 ||
    failed "the gtests against the bundle" "$log/gtest.log"
  tail -3 "$log/gtest.log"
}

report_bundle() {
  find "$bundle" -type f -exec wc -c {} + |
    awk -v dir="$bundle" '$2 != "total" { s += $1 } END { printf "bundle: %s, %.1f MB\n", dir, s / 1e6 }'
}
