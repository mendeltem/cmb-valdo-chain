#!/bin/bash
# wc3b Runde: Tests, 6 capped Maps (frst/gmm x k3/k5/k10), Ceiling-Analyse.
# Log: $CMB_ROOT/ergebnisse/wc3b_runde.log
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
cd $CMB_ROOT || exit 1
PY=${CMB_PYTHON:-python}
LOG=$CMB_ROOT/ergebnisse/wc3b_runde.log
: > "$LOG"

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

log "=== TESTS ==="
$PY -m cmb.tests.test_alt_candidates >>"$LOG" 2>&1
log "TESTS rc=$?"

for MODE in frst gmm; do
  for K in 3 5 10; do
    OUT="ergebnisse/turnier/wc3b-alt-${MODE}-k${K}"
    mkdir -p "$OUT"
    log "=== MAP ${MODE} k${K} -> ${OUT} ==="
    ( time $PY -m cmb.stage2.alt_candidates \
        --base ergebnisse/turnier/a03-aniso-e60-gd \
        --grid dev/gitter_d --mode "$MODE" --max-new "$K" --out "$OUT" \
      >>"$LOG" 2>&1 )
    log "MAP ${MODE} k${K} rc=$?"
  done
done

log "=== CEILING (7 runs) ==="
$PY -m cmb.analysis.candidate_ceiling \
  --run base=ergebnisse/turnier/a03-aniso-e60-gd \
  --run frst-k3=ergebnisse/turnier/wc3b-alt-frst-k3 \
  --run frst-k5=ergebnisse/turnier/wc3b-alt-frst-k5 \
  --run frst-k10=ergebnisse/turnier/wc3b-alt-frst-k10 \
  --run gmm-k3=ergebnisse/turnier/wc3b-alt-gmm-k3 \
  --run gmm-k5=ergebnisse/turnier/wc3b-alt-gmm-k5 \
  --run gmm-k10=ergebnisse/turnier/wc3b-alt-gmm-k10 \
  --grid dev/gitter_d --thresholds 0.15 \
  --json ergebnisse/analysis/wc3b_ceiling_capped.json >>"$LOG" 2>&1
log "CEILING rc=$?"
log "=== FERTIG ==="
