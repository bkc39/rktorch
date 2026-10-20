#!/usr/bin/env bash
#
#   scripts/pkg-build/memory-peaks.sh watch <log>     (until killed)
#   scripts/pkg-build/memory-peaks.sh report <log>
set -uo pipefail

usage="usage: memory-peaks.sh watch|report <log>"
log=${2:?$usage}

case ${1:?$usage} in
  watch)
    while true; do
      for id in $(docker ps -q --no-trunc); do
        name=$(docker inspect -f '{{.Name}}' "$id" 2>/dev/null) || continue
        for f in "/sys/fs/cgroup/system.slice/docker-$id.scope/memory.peak" \
                 "/sys/fs/cgroup/docker/$id/memory.peak"; do
          if [ -r "$f" ]; then
            echo "$(date -u +%H:%M:%S) ${name#/} ${id:0:12} $(cat "$f")" >>"$log"
            break
          fi
        done
      done
      sleep 2
    done
    ;;
  report)
    [ -s "$log" ] || { echo "no samples in $log"; exit 0; }
    awk '{ if (!($3 in first)) { first[$3] = $1; name[$3] = $2; order[n++] = $3 }
           if ($4 > peak[$3]) peak[$3] = $4 }
         END { for (i = 0; i < n; i++) { id = order[i]
                 printf "%s %-20s %s peak %.0f MB\n", first[id], name[id], id, peak[id] / 1048576 } }' "$log"
    ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
