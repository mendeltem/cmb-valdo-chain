#!/bin/bash
# wc3 Teil C (Fortsetzung): wc3-c ist bei C5 gestorben (kein rc, Prozess weg, ~05:33).
# Bereits fertig (wc3-c): C1-C4 (alle 9 alt-Ordner) + C2 ceiling (wc3_ceiling_alt.json)
#   + C5 t2s-frst999, swi-frst999, t2s-frst995.
# Offen: C5 frst995-swi, gmm-t2s, gmm-swi; VALDO alt -weich (frst999);
#   pooled boosting wc3_alt.json + wc3_base_boosting.json; paarweiser Vergleich.
# Variante frst999 = bester Ceiling-Gewinn je Kandidat (VALDO: F1 0.851->0.922,
#   33661 statt 278 Kandidaten). CPU only, 4 Threads.
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
PY=${CMB_PYTHON:-python}
V=$CMB_ROOT
MB=$CMB_PRIVATE
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$V"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "=== C5b: fehlende Kandidaten der alt-Varianten (privat) ==="
$PY -m cmb.stage2.candidates --predictions $MB/transfer/wc3-alt-frst995-cat-swi --manifest $MB/gitter/manifest_swi.json --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/wc3-alt-frst995-cat-swi-weich; log "cand swi frst995 rc=$?"
$PY -m cmb.stage2.candidates --predictions $MB/transfer/wc3-alt-gmm-cat-t2s --manifest $MB/gitter/manifest_t2star.json --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/wc3-alt-gmm-cat-t2s-weich; log "cand t2s gmm rc=$?"
$PY -m cmb.stage2.candidates --predictions $MB/transfer/wc3-alt-gmm-cat-swi --manifest $MB/gitter/manifest_swi.json --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/wc3-alt-gmm-cat-swi-weich; log "cand swi gmm rc=$?"

log "=== C6: VALDO alt-Kandidaten frst999 (--keep-csf) ==="
$PY -m cmb.stage2.candidates --predictions $V/ergebnisse/turnier/wc3-alt-frst999 --grid dev/gitter_d --keep-csf --out $V/ergebnisse/stage2/wc3-alt-frst999-weich; log "cand valdo frst999 rc=$?"

log "=== C7: pooled boosting, alt-Variante frst999 (alle 3 Kohorten) ==="
$PY -m cmb.stage2.pooled --cohort valdo=$V/ergebnisse/stage2/wc3-alt-frst999-weich t2star=$MB/stage2/wc3-alt-frst999-cat-t2s-weich swi=$MB/stage2/wc3-alt-frst999-cat-swi-weich --folds swi=$MB/gitter/manifest_swi_mix.json --models boosting --skip-own --tissue --json $MB/stage2/wc3_alt.json; log "pooled alt rc=$?"

log "=== C8: pooled boosting, Basis (wc3-base-*-weich mit centre_voxel, von D0) ==="
$PY -m cmb.stage2.pooled --cohort valdo=$V/ergebnisse/stage2/wc3-base-valdo-weich t2star=$MB/stage2/wc3-base-cat-t2s-weich swi=$MB/stage2/wc3-base-cat-swi-weich --folds swi=$MB/gitter/manifest_swi_mix.json --models boosting --skip-own --tissue --json $MB/stage2/wc3_base_boosting.json; log "pooled base rc=$?"

log "=== C9: paarweise alt vs. Basis (pooled+tissue boosting) ==="
$PY -m cmb.stage2.compare_runs --a $MB/stage2/wc3_alt.json "pooled+tissue boosting" --b $MB/stage2/wc3_base_boosting.json "pooled+tissue boosting" --cohorts valdo t2star swi; log "compare rc=$?"
log "=== TEIL C (Fortsetzung) fertig ==="
