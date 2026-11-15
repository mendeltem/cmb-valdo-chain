#!/usr/bin/env bash
# wk0a step 5 (GPU queue, SWI first): stage-1 a03-aniso-e60 on swi_v2 (same recipe and
# seed 42 as a03-aniso-e60-cat-swi; that run used manifest_swi.json, pure-SWI folds),
# candidates z40 (same settings as cat-swi-weich-z40: threshold 0.15, min 1 mm3,
# half 20 20 20, keep-csf), pooled stage 2 tset-plain seeds 0 and 1 with VALDO + T2*
# (old grids) + SWI v2, summary against the old chain -> <private>/maskfix/summary_swi.json.
#
# Fix 31.10.: stage 1 used manifest_swi.json = the OLD grids gitter/swi (unfilled masks); now
#   manifest_swi_v2.json = the same pure-SWI ids and folds, dirs -> gitter/swi_v2.
# Fix vs the 30.10. 23:18 draft:
#   [1] stage-1 manifest: manifest_swi.json (as cat-swi), NOT manifest_swi_mix.json.
#       mix folds the 7 twin cases into the T2* folds -> fold 0 would train on
#       one twin case and predict its own twin (data leak).
#   [2] candidates: --manifest manifest_swi_mix_v2.json (twin folds must match the
#       stage-2 --folds) + explicit --half 20 20 20 (z40; default is 20 20 10).
#   [3] fold_freeze checks: stage 1 against swi_stage1 (pure swi), stage 2 against
#       swi_stage2 (mix semantics) and t2star against t2star.
#   [4] summary: control = old pooled tset-plain (untagged), v2 runs tagged 'v2'.
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

# [0/6] smoke test removed 31.10.: turnier.py --probe reads ROOT/dev/split.json (VALDO ids) and ignores
# --manifest -> 0 cases, AssertionError after 2 s. Stage 1 below fails fast on its own if the v2 data are broken.

echo "[1/6] stage-1 a03-aniso-e60 auf swi_v2 (5 Falten, seed 42 wie cat-swi) $(date)"
$PY code/turnier.py --netz a03-aniso --epochen 60 --manifest $MB/gitter/manifest_swi_v2.json \
  --name cat-swi-v2 --basis $MB/transfer \
  > $MB/maskfix/gpu_stage1_swi_v2.log 2>&1 || { echo "FEHLER stage-1"; exit 1; }

echo "[2/6] fold_freeze check stage 1 (swi_stage1, reine SWI-Falten) $(date)"
$PY -m cmb.analysis.fold_freeze check \
  --frozen $MB/folds_frozen/swi_stage1.json --runs $MB/transfer/a03-aniso-e60-cat-swi-v2 \
  || { echo "FEHLER folds swi v2 stage1"; exit 1; }

echo "[3/6] Kandidaten z40 (gleiche Einstellungen wie cat-swi-weich-z40) $(date)"
$PY -m cmb.stage2.candidates --predictions $MB/transfer/a03-aniso-e60-cat-swi-v2 \
  --manifest $MB/gitter/manifest_swi_mix_v2.json \
  --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" \
  --keep-csf --half 20 20 20 --out $MB/stage2/a03-aniso-e60-cat-swi-v2-weich-z40 \
  > $MB/maskfix/gpu_cand_swi_v2.log 2>&1 || { echo "FEHLER candidates"; exit 1; }

echo "[4/6] fold_freeze check Kandidaten (swi_stage2, Mix-Falten) $(date)"
$PY -m cmb.analysis.fold_freeze check \
  --frozen $MB/folds_frozen/swi_stage2.json --candidates $MB/stage2/a03-aniso-e60-cat-swi-v2-weich-z40 \
  || { echo "FEHLER folds swi v2 stage2"; exit 1; }

echo "[5/6] gepoolte stage 2 tset-plain seeds 0 und 1 (VALDO + T2* alt + SWI v2) $(date)"
$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-union \
  t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-union \
  swi=$MB/stage2/a03-aniso-e60-cat-swi-v2-weich-z40 \
  --folds swi=$MB/gitter/manifest_swi_mix_v2.json --models tset-plain \
  --skip-own --tissue --seed-offset 0 --json $MB/maskfix/tset_v2_tset-plain_s0.json \
  > $MB/maskfix/gpu_tset_s0.log 2>&1 || { echo "FEHLER tset s0"; exit 1; }
$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-union \
  t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-union \
  swi=$MB/stage2/a03-aniso-e60-cat-swi-v2-weich-z40 \
  --folds swi=$MB/gitter/manifest_swi_mix_v2.json --models tset-plain \
  --skip-own --tissue --seed-offset 1 --json $MB/maskfix/tset_v2_tset-plain_s1.json \
  > $MB/maskfix/gpu_tset_s1.log 2>&1 || { echo "FEHLER tset s1"; exit 1; }

echo "[6/6] Summary v2 gegen die alte Kette -> $MB/maskfix/summary_swi.json $(date)"
# Kontrolle = 'pooled tset-plain' (alte Kandidaten, seeds 0+1); Arm = 'pooled tset-plain v2'
# (v2-Kandidaten, seeds 0+1) -- --tag macht aus den Modi der v2-Dateien '<mode> v2',
# sonst wuerden sie die Kontrolle ueberschreiben (later files win).
$PY -m cmb.stage2.tset_summary \
  --json $MB/stage2/tset_tset-plain_s0.json $MB/stage2/tset_tset-plain_s1.json \
        $MB/maskfix/tset_v2_tset-plain_s0.json $MB/maskfix/tset_v2_tset-plain_s1.json \
  --tag $MB/maskfix/tset_v2_tset-plain_s0.json=v2 $MB/maskfix/tset_v2_tset-plain_s1.json=v2 \
  --out $MB/maskfix/summary_swi.json > $MB/maskfix/gpu_summary.log 2>&1 \
|| { echo "FEHLER summary (tset_summary-Syntax, naechste Runde: von Hand zusammenstellen aus tset_v2_* und tset_tset-plain_*)"; exit 1; }
echo DONE_GPU_WK0A
