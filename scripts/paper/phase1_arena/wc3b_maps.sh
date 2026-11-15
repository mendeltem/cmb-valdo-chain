#!/usr/bin/env bash
# wc3b: sechs capped maps (MODE x K), CPU only
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -uo pipefail
cd $CMB_ROOT
PY=${CMB_PYTHON:-python}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
LOG=ergebnisse/analysis/wc3b_maps.log
: > "$LOG"
for MODE in frst gmm; do
  for K in 5 10; do
    OUT="ergebnisse/turnier/wc3b-alt-${MODE}-k${K}"
    echo "=== ${MODE} K=${K} -> ${OUT} $(date '+%H:%M:%S')" >> "$LOG"
    $PY -m cmb.stage2.alt_candidates --base ergebnisse/turnier/a03-aniso-e60-gd \
      --grid dev/gitter_d --mode $MODE --max-new $K --out "$OUT" >> "$LOG" 2>&1
    echo "rc=$? $(date '+%H:%M:%S')" >> "$LOG"
  done
done
echo DONE >> "$LOG"
