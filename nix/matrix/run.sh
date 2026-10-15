# `nix run .#matrix [-- <filter>...]`: every cell of MATRIX_CELLS for
# MATRIX_SYSTEM, or those whose "racket libtorch" contains a filter.
# A cell row: racket, libtorch, shim, then "skip" and a reason, or "build"
# and label=drv steps, built in order; a "run" step's program is executed.

logs=${MATRIX_LOG_DIR:-$(mktemp -d -t torch-matrix.XXXXXX)}
min_free=${MATRIX_MIN_FREE_GB:-30}
cell_timeout=${MATRIX_CELL_TIMEOUT:-6h}
nix_flags=(--no-link --print-out-paths --print-build-logs
  --max-jobs "${MATRIX_MAX_JOBS:-1}" --cores "${MATRIX_CORES:-4}"
  --max-silent-time 3600)
mkdir -p "$logs"

wanted() {
  [ "$#" -eq 1 ] && return 0
  local row=$1
  shift
  for pattern in "$@"; do
    case "$row" in *"$pattern"*) return 0 ;; esac
  done
  return 1
}

free_gb() {
  [ -r /proc/meminfo ] || return 0
  awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo
}

failure() {
  local log=$1 mark stage error
  mark=$(grep -n 'matrix-stage: ' "$log" | tail -1 | cut -d: -f1 || true)
  stage=$(grep -o 'matrix-stage: .*' "$log" | tail -1 | cut -d' ' -f2- || true)
  error=$(tail -n +"${mark:-1}" "$log" | grep -m1 -E '[Ee]rror|FAIL|[Ff]ailed|failure|Aborted' \
    | sed -E 's/^[^ ]*> //' | cut -c1-120 || true)
  echo "fail (${stage:-?}): ${error:-see log}"
}

run_cell() {
  local log=$1 step label drv out code
  shift
  for step in "$@"; do
    label=${step%%=*}
    drv=${step#*=}
    echo "matrix-stage: $label" >>"$log"
    if ! out=$(timeout "$cell_timeout" nix build "${nix_flags[@]}" "$drv^out" </dev/null 2>>"$log"); then
      failure "$log"
      return
    fi
    [ "$label" = run ] || continue
    code=0
    timeout "$cell_timeout" "$out/bin/$(ls "$out/bin")" </dev/null >>"$log" 2>&1 || code=$?
    case "$code" in
      0) ;;
      77) grep -o 'matrix-skip: .*' "$log" | tail -1 | sed 's/^matrix-skip/skip/'; return ;;
      *) failure "$log"; return ;;
    esac
  done
  echo pass
}

rows=()
failed=0
while IFS=$'\t' read -r racket libtorch shim mode rest; do
  wanted "$racket $libtorch" "$@" || continue
  name="${racket}-${libtorch// /-}-${shim}"
  log="$logs/$name.log"
  : >"$log"
  started=$SECONDS
  available=$(free_gb)
  echo "== $MATRIX_SYSTEM racket $racket, libtorch $libtorch, shim $shim" >&2
  if [ "$mode" = skip ]; then
    result="skip: $rest"
  elif [ -n "$available" ] && [ "$available" -lt "$min_free" ]; then
    result="skip: ${available} GB available < ${min_free}"
  else
    read -ra steps <<<"$rest"
    result=$(run_cell "$log" "${steps[@]}")
  fi
  case "$result" in fail*) failed=1 ;; esac
  echo "   $result ($((SECONDS - started))s, $log)" >&2
  rows+=("$(printf '%-14s %-6s %-18s %-5s %6ss  %s' \
    "$MATRIX_SYSTEM" "$racket" "$libtorch" "$shim" "$((SECONDS - started))" "$result")")
done <"$MATRIX_CELLS"

printf '%-14s %-6s %-18s %-5s %7s  %s\n' system racket libtorch shim time result
printf '%s\n' "${rows[@]}"
echo "logs: $logs"
exit "$failed"
