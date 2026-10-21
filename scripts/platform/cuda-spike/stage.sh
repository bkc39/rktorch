#!/usr/bin/env bash
# shellcheck disable=SC2016  # $ORIGIN is literal: the dynamic linker expands it
#
#   scripts/platform/cuda-spike/stage.sh <work-dir>
#
# Splits the .#cuda libraries into the packages cut.txt names, sets their
# runpaths, and zips each with the stock raco that racket.sh installed.
set -euo pipefail

work=$(realpath -m "${1:?usage: stage.sh <work-dir>}")
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
raco=$work/racket/bin/raco
[ -x "$raco" ] || { echo "stage: run racket.sh $work first" >&2; exit 2; }

paths=$(bash "$here/store-paths.sh")
declare -A source_dir=(
  [shim]=$(jq -r .shim <<<"$paths")/lib
  [libtorch]=$(jq -r .libtorch <<<"$paths")/lib
  [cudnn]=$(jq -r .cudnn <<<"$paths")/lib
  [nvrtc]=$(jq -r .nvrtc <<<"$paths")
)
patchelf=$(nix --max-jobs 1 --cores 4 build --inputs-from "$repo" --no-link --print-out-paths nixpkgs#patchelf)/bin/patchelf

# A package's libraries find each other through $ORIGIN, and the other
# packages' through their sibling directories under the same pkgs/ root.
declare -A runpath=(
  [torch-x86_64-linux]='$ORIGIN:$ORIGIN/../torch-cuda-libs-x86_64-linux:$ORIGIN/../torch-cudnn-x86_64-linux'
  [torch-cuda-libs-x86_64-linux]='$ORIGIN'
  [torch-cudnn-x86_64-linux]='$ORIGIN:$ORIGIN/../torch-cuda-libs-x86_64-linux'
)
packages=(torch-x86_64-linux torch-cuda-libs-x86_64-linux torch-cudnn-x86_64-linux)
system=" libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 ld-linux-x86-64.so.2 libstdc++.so.6 libgcc_s.so.1 libutil.so.1 libmvec.so.1 libz.so.1 libcuda.so.1 "

stage=$work/stage
zips=$work/zips
rm -rf "$stage" "$zips"
mkdir -p "$zips"
for package in "${packages[@]}"; do
  mkdir -p "$stage/$package"
  cp "$here/pkgs/$package/info.rkt" "$stage/$package/"
done

# NVIDIA's libraries and libtorch's are copied as shipped; the shim and its
# codecs are stripped.
install_lib() {
  local package=$1 file=$2 from=$3
  case $from in
    "${source_dir[libtorch]}"/* | "${source_dir[cudnn]}"/* | "${source_dir[nvrtc]}"/*)
      cp -L "$from" "$stage/$package/$file" ;;
    *) strip --strip-unneeded -o "$stage/$package/$file" "$from" ;;
  esac
  chmod 0644 "$stage/$package/$file"
  "$patchelf" --set-rpath "${runpath[$package]}" "$stage/$package/$file"
}

while read -r package file source; do
  case $package in '#'* | '') continue ;; esac
  install_lib "$package" "$file" "${source_dir[$source]}/$file"
done <"$here/cut.txt"

# The shim's non-CUDA closure (libsndfile and its codecs), from the store
# paths on each library's Nix runpath.
queue=("${source_dir[shim]}/libtorchrkt.so")
while [ "${#queue[@]}" -gt 0 ]; do
  from=${queue[0]}
  queue=("${queue[@]:1}")
  origin=$(readelf -d "$from" | sed -n 's/.*(RUNPATH).*\[\(.*\)\]/\1/p')
  for needed in $(readelf -d "$from" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
    case $system in *" $needed "*) continue ;; esac
    compgen -G "$stage/*/$needed" >/dev/null && continue
    found=
    for dir in ${origin//:/ }; do
      [ -e "$dir/$needed" ] && { found=$dir/$needed; break; }
    done
    [ -n "$found" ] || { echo "stage: $(basename "$from") NEEDs $needed, found nowhere" >&2; exit 1; }
    install_lib torch-x86_64-linux "$needed" "$found"
    queue+=("$found")
  done
done

failures=0
for lib in "$stage"/*/*.so*; do
  dir=$(dirname "$lib")
  rpath=$(readelf -d "$lib" | sed -n 's/.*(R\(UN\)\{0,1\}PATH).*\[\(.*\)\]/\2/p')
  for needed in $(readelf -d "$lib" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
    case $system in *" $needed "*) continue ;; esac
    resolved=
    for entry in ${rpath//:/ }; do
      [ -e "${entry/\$ORIGIN/$dir}/$needed" ] && { resolved=1; break; }
    done
    [ -n "$resolved" ] || { echo "  FAIL $(basename "$lib") NEEDs $needed, not on its runpath"; failures=$((failures + 1)); }
  done
  if grep -qaF /nix/store "$lib"; then
    echo "  note: $(basename "$lib") mentions $(grep -aoiE '/nix/store/[a-z0-9]{32}-[^/ ]+' "$lib" | sort -u | head -3 | tr '\n' ' ')"
  fi
done
[ "$failures" -eq 0 ] || { echo "stage: $failures unresolved NEEDED entries" >&2; exit 1; }
echo "stage: every NEEDED entry resolves inside the three packages or to a system library"

for package in "${packages[@]}"; do
  echo "== $package"
  bash "$here/../linux/audit-bundle.sh" "$stage/$package" >"$work/audit-$package.txt" || true
  grep -E 'FAIL|audit:' "$work/audit-$package.txt" |
    grep -vE 'runpath entry \$ORIGIN/\.\./|neither bundled nor a system library' || true
  start=$(date +%s)
  "$raco" pkg create --format zip --dest "$zips" "$stage/$package" >/dev/null
  echo "zipped in $(($(date +%s) - start)) s"
done

bash "$here/sizes.sh" "$work"
