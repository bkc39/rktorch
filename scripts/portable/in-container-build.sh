#!/usr/bin/env bash
# Runs inside quay.io/pypa/manylinux_2_28_x86_64 (glibc 2.28, gcc-toolset).
# Mounts: /src (repo, read-only), /libtorch (unpacked zip, read-only),
#         /work (scratch + output, read-write).
# Builds libsndfile and its codecs twice (static PIC archives, and shared
# libraries for the bundling comparison), then libtorchrkt against the
# static set with RUNPATH $ORIGIN.
set -euo pipefail

JOBS=${JOBS:-4}
W=/work
DL=$W/dl-src
mkdir -p "$DL" "$W/log"

fetch() {
  local url=$1 out=$DL/$(basename "$1")
  [[ -f $out ]] || curl -fsSL -o "$out" "$url"
  sha256sum "$out" >>"$W/log/sources.sha256"
}

fetch https://downloads.xiph.org/releases/ogg/libogg-1.3.6.tar.gz
fetch https://downloads.xiph.org/releases/vorbis/libvorbis-1.3.7.tar.gz
fetch https://downloads.xiph.org/releases/flac/flac-1.5.0.tar.xz
fetch https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz
fetch https://www.mpg123.de/download/mpg123-1.33.4.tar.bz2
fetch https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz
fetch https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz
# stb at the commit nixpkgs' stb-0-unstable-2023-01-29 pins.
mkdir -p "$W/stb/include/stb"
[[ -f $W/stb/include/stb/stb_image.h ]] || curl -fsSL -o "$W/stb/include/stb/stb_image.h" \
  https://raw.githubusercontent.com/nothings/stb/5736b15f7ea0ffb08dd38af21067c314d6a3aae9/stb_image.h
mkdir -p "$W/stb/lib/pkgconfig"
cat >"$W/stb/lib/pkgconfig/stb.pc" <<EOF
prefix=$W/stb
Name: stb
Description: stb single-file libraries
Version: 1
Cflags: -I$W/stb/include/stb
EOF

# build_codecs <prefix> <static|shared>
build_codecs() {
  local prefix=$1 kind=$2 flags
  if [[ $kind == static ]]; then
    flags="--enable-static --disable-shared --with-pic"
  else
    flags="--disable-static --enable-shared"
  fi
  [[ -f $prefix/.done ]] && return
  local b=$W/build-$kind
  rm -rf "$b" && mkdir -p "$b"
  export PKG_CONFIG_PATH=$prefix/lib/pkgconfig
  export CFLAGS="-O2 -fPIC" CPPFLAGS="-I$prefix/include" LDFLAGS="-L$prefix/lib"
  for t in libogg-1.3.6 libvorbis-1.3.7 flac-1.5.0 opus-1.5.2 mpg123-1.33.4 lame-3.100 libsndfile-1.2.2; do
    tar -C "$b" -xf "$DL"/$t.tar.*
    local extra=""
    case $t in
      flac-*) extra="--disable-programs --disable-examples --disable-doxygen-docs --disable-cpplibs" ;;
      opus-*) extra="--disable-doc --disable-extra-programs" ;;
      lame-*) extra="--disable-frontend --disable-decoder" ;;
      mpg123-*) extra="--disable-components --enable-libmpg123 --with-cpu=x86-64" ;;
      libsndfile-*) extra="--disable-full-suite --disable-sqlite --disable-alsa" ;;
    esac
    (cd "$b/$t" && ./configure --prefix="$prefix" $flags $extra >"$W/log/$kind-$t.configure.log" 2>&1 \
      && make -j"$JOBS" >"$W/log/$kind-$t.make.log" 2>&1 && make install >>"$W/log/$kind-$t.make.log" 2>&1) \
      || { echo "FAILED: $kind $t (see $W/log/$kind-$t.*.log)"; tail -20 "$W/log/$kind-$t".*.log; exit 1; }
  done
  grep -E "External|MPEG|Experimental" "$W/log/$kind-libsndfile-1.2.2.configure.log" || true
  touch "$prefix/.done"
}

time build_codecs "$W/deps-static" static
time build_codecs "$W/deps-shared" shared

# build_shim <static|shared>: link libsndfile and its codecs from that set.
build_shim() {
  local kind=$1 b=$W/shim-build-$1 o=$W/out/$1 libs
  export PKG_CONFIG_PATH=$W/deps-$kind/lib/pkgconfig:$W/stb/lib/pkgconfig
  if [[ $kind == static ]]; then
    # glibc 2.28 keeps pthread in libpthread (merged into libc only in 2.34),
    # and flac.pc does not list it.
    libs="$(pkg-config --static --libs sndfile) -lpthread"
  else
    libs=""
  fi
  echo "shim ($kind sndfile) extra link line: $libs"
  # --no-undefined: a missing codec symbol fails the link, not the dlopen.
  # --exclude-libs,ALL: archived codec symbols stay local to the shim, so a
  # host libsndfile loaded into the same process cannot interpose on them.
  cmake -S /src/cpp -B "$b" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_PREFIX_PATH=/libtorch \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
    "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--no-undefined -Wl,--exclude-libs,ALL -Wl,--enable-new-dtags -Wl,--as-needed" \
    "-DCMAKE_CXX_STANDARD_LIBRARIES=$libs" \
    >"$W/log/shim-$kind-configure.log" 2>&1 || { tail -40 "$W/log/shim-$kind-configure.log"; exit 1; }
  time cmake --build "$b" --target torchrkt -j "$JOBS" >"$W/log/shim-$kind-build.log" 2>&1 \
    || { grep -m20 -E "error|undefined" "$W/log/shim-$kind-build.log"; exit 1; }
  mkdir -p "$o"
  strip --strip-unneeded -o "$o/libtorchrkt.so" "$b/libtorchrkt.so"
}

build_shim static
build_shim shared
gcc --version | head -1 >"$W/out/toolchain.txt"
ldd --version | head -1 >>"$W/out/toolchain.txt"
ls -lR "$W/out"
