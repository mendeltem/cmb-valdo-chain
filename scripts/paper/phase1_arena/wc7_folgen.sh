#!/usr/bin/env bash
# wc7: wartet auf die Kandidaten-Kette, prueft Falten, reihst die beiden GPU-Laeufe ein,
# wartet dann (Rundenuebergabe) und macht am Ende das Summary.
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
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$V"

echo "[1] warte auf wc7-cand-union (max 3600 s) ..."
rc_wait=$(waitjob wc7-cand-union --max 3600 > /dev/null 2>&1; echo $?)
echo "rc_wait=$rc_wait (3 = noch laeuft -> Rundenuebergabe, naechste Runde setzt fort)"
if [ "$rc_wait" -ne 0 ]; then echo "STOPP: Kette nicht fertig oder Fehler"; exit 1; fi

echo "[2] fold_freeze check (3 Kohorten)"
$PY -m cmb.analysis.fold_freeze check --frozen dev/folds_frozen/valdo.json --candidates ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-union || { echo "FEHLER folds valdo"; exit 1; }
$PY -m cmb.analysis.fold_freeze check --frozen $MB/folds_frozen/t2star.json --candidates $MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-union || { echo "FEHLER folds t2star"; exit 1; }
$PY -m cmb.analysis.fold_freeze check --frozen $MB/folds_frozen/swi_stage1.json --candidates $MB/stage2/a03-aniso-e60-cat-swi-weich-z40-union || { echo "FEHLER folds swi"; exit 1; }

echo "[3] Queue: tset-plain und tset-ctx auf den -union-Kandidaten"
queue --einreihen --vram 12 -- bash -c "cd $V && $PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-union t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-union swi=$MB/stage2/a03-aniso-e60-cat-swi-weich-z40-union --folds swi=$MB/gitter/manifest_swi_mix.json --models tset-plain --skip-own --tissue --seed-offset 0 --json $MB/stage2/tset_union_tset-plain_s0.json > $MB/stage2/tset_union_tset-plain_s0.log 2>&1" || { echo "FEHLER einreihen tset-plain"; exit 1; }
queue --einreihen --vram 12 -- bash -c "cd $V && $PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-union t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-union swi=$MB/stage2/a03-aniso-e60-cat-swi-weich-z40-union --folds swi=$MB/gitter/manifest_swi_mix.json --models tset-ctx --skip-own --tissue --seed-offset 0 --json $MB/stage2/tset_union_tset-ctx_s0.json > $MB/stage2/tset_union_tset-ctx_s0.log 2>&1" || { echo "FEHLER einreihen tset-ctx"; exit 1; }

echo "[4] warte auf beide JSONs (max 3300 s; Queue laeuft hinter einem anderen Projekt, Antreiber arbeitet ab) ..."
t=0
while [ $t -lt 3300 ]; do
  if [ -s $MB/stage2/tset_union_tset-plain_s0.json ] && [ -s $MB/stage2/tset_union_tset-ctx_s0.json ]; then break; fi
  sleep 60; t=$((t+60))
done
[ -s $MB/stage2/tset_union_tset-plain_s0.json ] || { echo "STOPP: tset_union_tset-plain_s0.json fehlt noch (Rundenuebergabe)"; exit 3; }
[ -s $MB/stage2/tset_union_tset-ctx_s0.json ] || { echo "STOPP: tset_union_tset-ctx_s0.json fehlt noch (Rundenuebergabe)"; exit 3; }

echo "[5] Summary (Regeln v2, Kontrolle: pooled tset-plain)"
$PY -m cmb.stage2.tset_summary \
  --json $MB/stage2/tset_tset-plain_s0.json $MB/stage2/tset_tset-ctx_s0.json \
  --tag $MB/stage2/tset_union_tset-plain_s0.json=union $MB/stage2/tset_union_tset-ctx_s0.json=union \
  --control "pooled tset-plain" \
  --out $MB/stage2/tset_summary_union_round1.json || { echo "FEHLER summary"; exit 1; }

echo "[6] Entscheidung (nur Zitausgabe, keine Reinterpretation der Regeln)"
grep -E '"arm"|"decision"|"delta_all"|"shrunk"' $MB/stage2/tset_summary_union_round1.json | head -40

echo "=== wc7 FOLGEN FERTIG ==="
