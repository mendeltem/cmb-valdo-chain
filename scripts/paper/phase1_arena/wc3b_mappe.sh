#!/usr/bin/env bash
# wc3b: CMB ARENA 2 -- capped candidate source (VALDO dev).
# 1) Tests, 2) sechs gecappte Karten (frst/gmm x K 3/5/10), 3) ceiling-Hilfe dumpen.
# Log: ergebnisse/analysis/wc3b_mappe.log  -- rc in wc3b_mappe.rc
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
cd $CMB_ROOT
PY=${CMB_PYTHON:-python}
LOG=ergebnisse/analysis/wc3b_mappe.log
: > "$LOG"
echo "=== $(date '+%F %T') START ===" >> "$LOG"

echo "--- tests test_alt_candidates ---" | tee -a "$LOG"
$PY -m cmb.tests.test_alt_candidates >> "$LOG" 2>&1
RC=$?
echo "tests rc=$RC" | tee -a "$LOG"

for MODE in frst gmm; do
  for K in 3 5 10; do
    OUT="ergebnisse/turnier/wc3b-alt-${MODE}-k${K}"
    echo "--- map $MODE k$K start $(date '+%T') ---" | tee -a "$LOG"
    $PY -m cmb.stage2.alt_candidates \
      --base ergebnisse/turnier/a03-aniso-e60-gd \
      --grid dev/gitter_d \
      --mode "$MODE" --max-new "$K" \
      --out "$OUT" >> "$LOG" 2>&1
    RC=$?
    echo "map $MODE k$K rc=$RC end $(date '+%T')" | tee -a "$LOG"
  done
done

echo "--- candidate_ceiling --help ---" | tee -a "$LOG"
$PY -m cmb.analysis.candidate_ceiling --help >> "$LOG" 2>&1
echo "help rc=$?" | tee -a "$LOG"

echo "=== $(date '+%F %T') ENDE ===" | tee -a "$LOG"
echo OK > $CMB_ROOT/ergebnisse/analysis/wc3b_mappe.rc
