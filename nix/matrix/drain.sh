now() { echo "${EPOCHREALTIME//[!0-9]/}"; }

live() {
  kill -0 -- -"$1" 2>/dev/null || return 1
  if [ -r /proc/self/stat ]; then
    [ -n "$(cat /proc/[0-9]*/stat 2>/dev/null \
      | awk -v g="$1" '{ sub(/.*\) /, "") } $3 == g && $1 != "Z" { print; exit }')" ]
  elif command -v ps >/dev/null; then
    ps -A -o pgid= -o stat= | awk -v g="$1" '$1 == g && $2 !~ /^Z/ { n++ } END { exit !n }'
  fi
}

settle() {
  local until=$(($(now) + $2))
  while live "$1" && [ "$(now)" -lt "$until" ]; do
    sleep 0.1
  done
}

drain() {
  kill -TERM -- -"$1" 2>/dev/null || return 0
  settle "$1" 5000000
  kill -KILL -- -"$1" 2>/dev/null || true
  settle "$1" 2000000
  wait "$1" 2>/dev/null || true
}
