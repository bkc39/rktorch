#!/usr/bin/env bash
# The bundling alternative to linking libsndfile statically: copy the shared
# libsndfile and the codecs it NEEDs, stripped, with RUNPATH $ORIGIN.
#   scripts/portable/stage-sndfile-shared.sh <deps-shared/lib> <out-dir>
# Needs patchelf on PATH.
set -euo pipefail

src=$1 out=$2
mkdir -p "$out"
for f in libsndfile.so.1 libFLAC.so.14 libvorbisenc.so.2 libvorbis.so.0 \
         libopus.so.0 libogg.so.0 libmpg123.so.0 libmp3lame.so.0; do
  strip --strip-unneeded -o "$out/$f" "$src/$f"
  patchelf --set-rpath '$ORIGIN' "$out/$f"
done
du -cb "$out"/lib*.so* | tail -1
