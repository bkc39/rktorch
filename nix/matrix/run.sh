logs=${MATRIX_LOG_DIR:-$(mktemp -d -t torch-matrix.XXXXXX)}
min_free=${MATRIX_MIN_FREE_GB:-30}
cell_timeout=${MATRIX_CELL_TIMEOUT:-6h}
nix_flags=(--no-link --print-out-paths --print-build-logs
  --max-jobs "${MATRIX_MAX_JOBS:-1}" --cores "${MATRIX_CORES:-4}"
  --max-silent-time 3600)
mkdir -p "$logs"
set -m

[[ $cell_timeout =~ ^([0-9]+)([smhd]?)$ ]] || {
  echo "torch-matrix: MATRIX_CELL_TIMEOUT is N[smhd], got $cell_timeout" >&2
  exit 2
}
case "${BASH_REMATCH[2]}" in
  d) budget=$((BASH_REMATCH[1] * 86400000000)) ;;
  h) budget=$((BASH_REMATCH[1] * 3600000000)) ;;
  m) budget=$((BASH_REMATCH[1] * 60000000)) ;;
  *) budget=$((BASH_REMATCH[1] * 1000000)) ;;
esac

now() { echo "${EPOCHREALTIME//[!0-9]/}"; }

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
  case "$MATRIX_SYSTEM" in
    *-linux) awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo ;;
    *-darwin) vm_stat | awk '
      /page size of/ { size = $8 }
      /^Pages (free|inactive|speculative):/ { pages += $NF }
      END { if (size) print int(pages * size / 1073741824) }' ;;
  esac
}

failure() {
  local log=$1 mark stage error
  mark=$(grep -n 'matrix-stage: ' "$log" | tail -1 | cut -d: -f1 || true)
  stage=$(grep -o 'matrix-stage: .*' "$log" | tail -1 | cut -d' ' -f2- || true)
  error=$(tail -n +"${mark:-1}" "$log" | awk '
    { sub(/^[^ ]*> /, "") }
    /^raco test: / { prev = ""; next }
    /^known failure / { next }
    check != "" && NF && rack == "" { rack = check " " name ": " $0 }
    /^(ERROR|FAILURE)$/ { check = $0; name = prev; next }
    /^\[  FAILED  \] / && gtest == "" { gtest = $0 }
    /: error:/ && compile == "" { compile = $0 }
    /^error:/ { last = $0 }
    { prev = $0 }
    END {
      if (gtest != "") print gtest
      else if (compile != "") print compile
      else if (rack != "") print rack
      else if (last != "") print last
    }' | cut -c1-160 || true)
  echo "fail (${stage:-?}): ${error:-see log}"
}

drain() {
  local tries=0
  kill -TERM -- -"$1" 2>/dev/null || return 0
  wait "$1" 2>/dev/null || true
  while kill -0 -- -"$1" 2>/dev/null; do
    tries=$((tries + 1))
    if [ "$tries" -ge 50 ]; then
      kill -KILL -- -"$1" 2>/dev/null || true
      return 0
    fi
    sleep 0.1
  done
}

group=
step() {
  local code=0 stdout=$1 stderr=$2
  shift 2
  "$@" </dev/null >>"$stdout" 2>>"$stderr" &
  group=$!
  wait "$group" || code=$?
  return "$code"
}

deadline=
remaining() {
  local left=$((deadline - $(now)))
  [ "$left" -gt 0 ] || return 1
  echo "$((left / 1000000)).$(printf '%06d' $((left % 1000000)))"
}

stage=
result=
run_cell() {
  local log=$1 s label drv out left code
  shift
  deadline=$(($(now) + budget))
  for s in "$@"; do
    label=${s%%=*}
    drv=${s#*=}
    stage=$label
    echo "matrix-stage: $label" >>"$log"
    code=0
    group=
    : >"$log.out"
    if ! left=$(remaining); then
      code=124
    else
      step "$log.out" "$log" timeout --foreground -k 30 "$left" \
        nix build "${nix_flags[@]}" "$drv^out" || code=$?
    fi
    out=$(cat "$log.out")
    if [ "$code" != 0 ] || [ "$label" != run ]; then
      :
    elif ! left=$(remaining); then
      code=124
    else
      step "$log" "$log" timeout --foreground -k 30 "$left" \
        "$out/bin/$(ls "$out/bin")" || code=$?
    fi
    case "$code" in
      0) ;;
      124)
        [ -z "$group" ] || drain "$group"
        result="fail ($label): timed out after $cell_timeout"
        return
        ;;
      77)
        result=$(grep -o 'matrix-skip: .*' "$log" | tail -1 | sed 's/^matrix-skip/skip/')
        return
        ;;
      *)
        result=$(failure "$log")
        return
        ;;
    esac
  done
  result=pass
}

rows=()
row() {
  rows+=("$(printf '%-14s %-6s %-18s %-5s %6ss  %s' "$MATRIX_SYSTEM" "$@")")
}

table() {
  printf '%-14s %-6s %-18s %-5s %7s  %s\n' system racket libtorch shim time result
  [ "${#rows[@]}" -eq 0 ] || printf '%s\n' "${rows[@]}"
  echo "logs: $logs"
}

running=0
stop() {
  trap '' INT TERM
  for g in $(jobs -p); do drain "$g"; done
  [ "$running" = 0 ] \
    || row "$racket" "$libtorch" "$shim" "$((SECONDS - started))" "fail (${stage:-?}): interrupted"
  table
  exit "$1"
}
trap 'stop 130' INT
trap 'stop 143' TERM

passed=0
failed=0
guarded=0
while IFS=$'\t' read -r racket libtorch shim mode rest; do
  wanted "$racket $libtorch" "$@" || continue
  name="${racket}-${libtorch// /-}-${shim}"
  log="$logs/$name.log"
  : >"$log"
  started=$SECONDS
  stage=
  note=
  available=$(free_gb 2>/dev/null || true)
  echo "== $MATRIX_SYSTEM racket $racket, libtorch $libtorch, shim $shim" >&2
  if [ "$mode" = skip ]; then
    result="skip: $rest"
  elif [ -n "$available" ] && [ "$available" -lt "$min_free" ]; then
    result="skip: ${available} GB available < ${min_free}"
  else
    [ -n "$available" ] || note=" (free memory unread, guard off)"
    read -ra steps <<<"$rest"
    running=1
    run_cell "$log" "${steps[@]}"
    running=0
  fi
  case "$mode:$result" in
    *:pass) passed=$((passed + 1)) ;;
    *:fail*) failed=1 ;;
    build:skip*) guarded=1 ;;
  esac
  echo "   $result$note ($((SECONDS - started))s, $log)" >&2
  row "$racket" "$libtorch" "$shim" "$((SECONDS - started))" "$result$note"
done <"$MATRIX_CELLS"

table
if [ "$failed" = 1 ]; then
  exit 1
elif [ "$passed" = 0 ] && [ "$guarded" = 1 ]; then
  echo "torch-matrix: no cell passed, and a resource guard skipped some" >&2
  exit 2
fi
