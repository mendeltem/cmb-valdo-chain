#!/usr/bin/env bash
# wc7 (naechste Runde): wartet auf die beiden Union-JSONs und macht das Summary (Regeln v2).
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
PY=${CMB_PYTHON:-python}
V=$CMB_ROOT
MB=$CMB_PRIVATE
cd "$V"

echo "[0] warte auf beide JSONs (max 3500 s; Queue-Laeufe [373]/[374] laufen nach Rundenende, je ~36-40 min) ..."
t=0
while [ $t -lt 3500 ]; do
  if [ -s $MB/stage2/tset_union_tset-plain_s0.json ] && [ -s $MB/stage2/tset_union_tset-ctx_s0.json ]; then break; fi
  sleep 60; t=$((t+60))
done
[ -s $MB/stage2/tset_union_tset-plain_s0.json ] || { echo "STOPP: tset_union_tset-plain_s0.json fehlt noch"; exit 3; }
[ -s $MB/stage2/tset_union_tset-ctx_s0.json ] || { echo "STOPP: tset_union_tset-ctx_s0.json fehlt noch"; exit 3; }

echo "[1] Summary (Regeln v2, Kontrolle: pooled tset-plain)"
$PY -m cmb.stage2.tset_summary \
  --json $MB/stage2/tset_tset-plain_s0.json $MB/stage2/tset_tset-ctx_s0.json \
  --tag $MB/stage2/tset_union_tset-plain_s0.json=union $MB/stage2/tset_union_tset-ctx_s0.json=union \
  --control "pooled tset-plain" \
  --out $MB/stage2/tset_summary_union_round1.json || { echo "FEHLER summary"; exit 1; }

echo "[2] Zitausgabe der Entscheidungen"
grep -E '"arm"|"decision"|"delta_all"|"shrunk"' $MB/stage2/tset_summary_union_round1.json | head -40
echo "=== wc7 SUMMARY FERTIG ==="
