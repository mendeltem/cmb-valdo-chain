#!/usr/bin/env bash
# wc7: wartet auf tset_union_tset-ctx_s1.json (Queue [375], laeuft am Rundenende),
# dann Final-Summary ueber alle Seeds (Regeln v2, Kontrolle pooled tset-plain).
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
PY=${CMB_PYTHON:-python}
MB=$CMB_PRIVATE
cd $CMB_ROOT

echo "[1] warte auf tset_union_tset-ctx_s1.json (max 4800 s) ..."
t=0
while [ $t -lt 4800 ]; do
  if [ -s $MB/stage2/tset_union_tset-ctx_s1.json ]; then break; fi
  sleep 60; t=$((t+60))
done
[ -s $MB/stage2/tset_union_tset-ctx_s1.json ] || { echo "STOPP: tset_union_tset-ctx_s1.json fehlt noch (Rundenuebergabe)"; exit 3; }

echo "[2] Final-Summary ueber alle Seeds (Regeln v2)"
$PY -m cmb.stage2.tset_summary \
  --json $MB/stage2/tset_tset-plain_s0.json $MB/stage2/tset_tset-ctx_s0.json $MB/stage2/tset_tset-plain_s1.json \
  --tag $MB/stage2/tset_union_tset-plain_s0.json=union \
       $MB/stage2/tset_union_tset-ctx_s0.json=union \
       $MB/stage2/tset_union_tset-ctx_s1.json=union \
  --control "pooled tset-plain" \
  --out $MB/stage2/tset_summary_union_final.json || { echo "FEHLER summary final"; exit 1; }

echo "[3] Zitausgabe der Entscheidungen"
grep -E '"arm"|"decision"|"delta_all"|"shrunk"' $MB/stage2/tset_summary_union_final.json | head -40
echo "=== wc7 SEED1 FOLGEN FERTIG ==="
