#!/usr/bin/env bash
#
#   bash /src/scripts/platform/linux/in-container.sh   (run by build.sh)
set -euo pipefail

jobs=${JOBS:-4}
here=/src/scripts/platform/linux
work=/work
dl=$work/dl
log=$work/log
deps=$work/deps
libtorch=$work/libtorch/libtorch
bundle=$work/bundle
mkdir -p "$dl" "$log"

failed() {
  echo "build: $1 failed; see $2" >&2
  tail -30 "$2" >&2
  exit 1
}

current() {
  [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]
}

declare -A pinned
while read -r sum file url; do
  case $sum in '' | \#*) continue ;; esac
  if [ ! -f "$dl/$file" ]; then
    curl -fsSL -o "$dl/$file.part" "$url"
    mv "$dl/$file.part" "$dl/$file"
  fi
  echo "$sum  $dl/$file" | sha256sum --check --quiet -
  pinned[$file]=$sum
  case $file in libtorch-*.zip) libtorch_zip=$file ;; esac
done <"$here/sources.txt"

if ! current "$work/libtorch/.pinned" "${pinned[$libtorch_zip]}"; then
  rm -rf "$work/libtorch"
  unzip -q -d "$work/libtorch" "$dl/$libtorch_zip"
  echo "${pinned[$libtorch_zip]}" >"$work/libtorch/.pinned"
fi

mkdir -p "$work/stb/include/stb" "$work/stb/lib/pkgconfig"
cp "$dl/stb_image.h" "$work/stb/include/stb/"
cat >"$work/stb/lib/pkgconfig/stb.pc" <<EOF
Name: stb
Description: stb single-file libraries
Version: 1
Cflags: -I$work/stb/include/stb
EOF

codecs=(libogg-1.3.6.tar.gz libvorbis-1.3.7.tar.gz flac-1.5.0.tar.xz opus-1.5.2.tar.gz
        mpg123-1.33.4.tar.bz2 lame-3.100.tar.gz libsndfile-1.2.2.tar.xz)
codec_cflags="-O2 -fPIC"
declare -A configure_flags=(
  [flac-1.5.0.tar.xz]="--disable-programs --disable-examples --disable-doxygen-docs --disable-cpplibs"
  [opus-1.5.2.tar.gz]="--disable-doc --disable-extra-programs"
  [lame-3.100.tar.gz]="--disable-frontend --disable-decoder"
  [mpg123-1.33.4.tar.bz2]="--disable-components --enable-libmpg123 --with-cpu=x86-64"
  [libsndfile-1.2.2.tar.xz]="--disable-full-suite --disable-sqlite --disable-alsa"
)
codecs_key=$(echo "CFLAGS=$codec_cflags"
             for tarball in "${codecs[@]}"; do
               echo "${pinned[$tarball]} $tarball ${configure_flags[$tarball]-}"
             done)
if ! current "$deps/.pinned" "$codecs_key"; then
  rm -rf "$deps" "$work/codec-build"
  mkdir -p "$work/codec-build"
  export PKG_CONFIG_PATH=$deps/lib/pkgconfig
  export CFLAGS=$codec_cflags CPPFLAGS="-I$deps/include" LDFLAGS="-L$deps/lib"
  for tarball in "${codecs[@]}"; do
    codec=${tarball%.tar.*}
    tar -C "$work/codec-build" -xf "$dl/$tarball"
    read -ra extra <<<"${configure_flags[$tarball]-}"
    (cd "$work/codec-build/$codec" &&
       ./configure --prefix="$deps" --disable-static --enable-shared "${extra[@]}" &&
       make -j"$jobs" && make install) >"$log/$codec.log" 2>&1 ||
      failed "$codec" "$log/$codec.log"
  done
  sndfile_log=$log/${codecs[-1]%.tar.*}.log
  for line in 'External FLAC/Ogg/Vorbis/Opus : [.]* yes' 'External MPEG Lame/MPG123 : [.]* yes'; do
    grep -q "$line" "$sndfile_log" || failed "libsndfile without its codecs" "$sndfile_log"
  done
  unset PKG_CONFIG_PATH CFLAGS CPPFLAGS LDFLAGS
  echo "$codecs_key" >"$deps/.pinned"
fi

gtest=googletest-1.17.0
if ! current "$work/gtest/.pinned" "${pinned[$gtest.tar.gz]}"; then
  rm -rf "$work/gtest" "$work/gtest-build" "$work/$gtest"
  tar -C "$work" -xf "$dl/$gtest.tar.gz"
  (cmake -S "$work/$gtest" -B "$work/gtest-build" -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_GMOCK=OFF -DCMAKE_INSTALL_PREFIX="$work/gtest" &&
     cmake --build "$work/gtest-build" -j "$jobs" &&
     cmake --install "$work/gtest-build") >"$log/googletest.log" 2>&1 ||
    failed googletest "$log/googletest.log"
  echo "${pinned[$gtest.tar.gz]}" >"$work/gtest/.pinned"
fi

shim_key=$(printf '%s\n' "${pinned[$libtorch_zip]}" "$codecs_key" "${pinned[$gtest.tar.gz]}")
if ! current "$work/shim-build/.inputs" "$shim_key"; then
  rm -rf "$work/shim-build"
  mkdir -p "$work/shim-build"
  echo "$shim_key" >"$work/shim-build/.inputs"
fi

# --exclude-libs keeps gcc-toolset's static libstdc++_nonshared pieces out of
# the shim's exports.
export PKG_CONFIG_PATH=$deps/lib/pkgconfig:$work/stb/lib/pkgconfig
export LD_LIBRARY_PATH=$deps/lib:$libtorch/lib
(cmake -S /src/cpp -B "$work/shim-build" -G "Unix Makefiles" \
   -DCMAKE_BUILD_TYPE=Release \
   -DBUILD_TESTING=ON \
   "-DCMAKE_PREFIX_PATH=$libtorch;$work/gtest" \
   -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
   '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
   "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--no-undefined -Wl,--exclude-libs,ALL -Wl,--enable-new-dtags -Wl,--as-needed" &&
   cmake --build "$work/shim-build" -j "$jobs") >"$log/shim.log" 2>&1 ||
  failed "the shim" "$log/shim.log"
unset LD_LIBRARY_PATH

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

LD_LIBRARY_PATH=$bundle "$work/shim-build/torchrkt_tests" --gtest_brief=1 \
  >"$log/gtest.log" 2>&1 || failed "the gtests against the bundle" "$log/gtest.log"
tail -3 "$log/gtest.log"
du -sb "$bundle" | awk '{ printf "bundle: %s, %.1f MB\n", $2, $1 / 1e6 }'
