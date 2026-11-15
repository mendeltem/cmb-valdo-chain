#!/bin/bash
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
V=$CMB_ROOT; MB=$CMB_PRIVATE
echo "=== alt variant JSONs ==="
for v in frst999 frst995 gmm; do
  for d in $V/ergebnisse/turnier/wc3-alt-$v $MB/transfer/wc3-alt-$v-cat-t2s $MB/transfer/wc3-alt-$v-cat-swi; do
    [ -s "$d/alt_candidates.json" ] && echo "OK   $d" || echo "MISS $d"
  done
done
echo "=== private cohort candidate counts (last lines) ==="
for d in $MB/stage2/wc3-alt-frst999-cat-t2s-weich $MB/stage2/wc3-alt-frst999-cat-swi-weich $V/ergebnisse/stage2/wc3-alt-frst999-weich; do
  echo "--- $d"; tail -2 "$d/candidates.json.log" 2>/dev/null || ls "$d" | head
done
echo "=== run times ==="
for f in wc3-a wc3-c wc3-c2 wc3-c3 wc3-d2 wc3-d; do
  L=$CMB_LOGS/$f.log
  [ -f "$L" ] && echo "$f: $(head -1 $L | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1) -> $(tail -1 $L | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)"
done
