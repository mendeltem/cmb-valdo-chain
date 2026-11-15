#!/usr/bin/env bash
# wc3b Probelauf: tests + 1 Karte (frst k3), Zeit messen.
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
echo "=== $(date '+%F %T') PROBELAUF START ===" >> "$LOG"

echo "--- tests test_alt_candidates ---" | tee -a "$LOG"
$PY -m cmb.tests.test_alt_candidates >> "$LOG" 2>&1
RC=$?
echo "tests rc=$RC" | tee -a "$LOG"

echo "--- map frst k3 start $(date '+%T') ---" | tee -a "$LOG"
T0=$(date +%s)
$PY -m cmb.stage2.alt_candidates \
  --base ergebnisse/turnier/a03-aniso-e60-gd \
  --grid dev/gitter_d \
  --mode frst --max-new 3 \
  --out ergebnisse/turnier/wc3b-alt-frst-k3 >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
echo "map frst k3 rc=$RC dauer $((T1-T0)) s end $(date '+%T')" | tee -a "$LOG"

echo "--- candidate_ceiling --help ---" | tee -a "$LOG"
$PY -m cmb.analysis.candidate_ceiling --help >> "$LOG" 2>&1
echo "=== $(date '+%F %T') PROBELAUF ENDE ===" | tee -a "$LOG"
echo "map_rc=$RC" > $CMB_ROOT/ergebnisse/analysis/wc3b_probelauf.rc
