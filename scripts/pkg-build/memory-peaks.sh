#!/usr/bin/env bash
#
#   scripts/pkg-build/memory-peaks.sh watch|report <log>
set -uo pipefail

usage="usage: memory-peaks.sh watch|report <log>"
log=${2:?$usage}

read_or_zero() { if [ -r "$1" ]; then cat "$1"; else echo 0; fi; }

case ${1:?$usage} in
  watch)
    while true; do
      for id in $(docker ps -q --no-trunc); do
        name=$(docker inspect -f '{{.Name}}' "$id" 2>/dev/null) || continue
        for cg in "/sys/fs/cgroup/system.slice/docker-$id.scope" "/sys/fs/cgroup/docker/$id"; do
          if [ -r "$cg/memory.peak" ]; then
            anon=$(awk '$1 == "anon" { print $2 }' "$cg/memory.stat")
            kills=$(awk '$1 == "oom_kill" { print $2 }' "$cg/memory.events")
            echo "$(date -u +%H:%M:%S) ${name#/} ${id:0:12} $(cat "$cg/memory.peak")" \
                 "${anon:-0} $(read_or_zero "$cg/memory.swap.peak") ${kills:-0}" >>"$log"
            break
          fi
        done
      done
      sleep 2
    done
    ;;
  report)
    [ -s "$log" ] || { echo "no samples in $log"; exit 0; }
    echo "peak counts page cache, which the kernel reclaims at the cap; anon is the largest sample"
    awk 'function mb(b) { return b / 1048576 }
         function up(a, k, v) { if (v > a[k]) a[k] = v }
         { if (!($3 in first)) { first[$3] = $1; name[$3] = $2; order[n++] = $3 }
           up(peak, $3, $4); up(anon, $3, $5); up(swap, $3, $6); up(kills, $3, $7) }
         END { for (i = 0; i < n; i++) { id = order[i]
                 printf "%s %-20s %s peak %4.0f MB  anon %4.0f MB  swap %4.0f MB  oom-kills %d\n",
                        first[id], name[id], id, mb(peak[id]), mb(anon[id]), mb(swap[id]), kills[id] } }' "$log"
    ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
