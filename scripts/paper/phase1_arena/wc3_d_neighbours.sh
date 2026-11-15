#!/bin/bash
# wc3 Teil D: Neighbour-Features (Proxy fuer das Set-Modul).
# Die alten -weich-Ordner haben kein 'centre_voxel' (Code-Erweiterung 2026-10-25),
# deshalb Basis-Kandidatensaetze unter wc3-Präfix NEU aufbauen, dann pooled --neighbours.
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

log "=== D0: Basis-Kandidaten neu (mit centre_voxel) ==="
$PY -m cmb.stage2.candidates --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --keep-csf --out ergebnisse/stage2/wc3-base-valdo-weich; log "cand valdo rc=$?"
$PY -m cmb.stage2.candidates --predictions $MB/transfer/a03-aniso-e60-cat-t2s --manifest $MB/gitter/manifest_t2star.json --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/wc3-base-cat-t2s-weich; log "cand t2s rc=$?"
$PY -m cmb.stage2.candidates --predictions $MB/transfer/a03-aniso-e60-cat-swi --manifest $MB/gitter/manifest_swi.json --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/wc3-base-cat-swi-weich; log "cand swi rc=$?"

log "=== D1: pooled --neighbours (logistic boosting) ==="
$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/wc3-base-valdo-weich t2star=$MB/stage2/wc3-base-cat-t2s-weich swi=$MB/stage2/wc3-base-cat-swi-weich --folds swi=$MB/gitter/manifest_swi_mix.json --models logistic boosting --skip-own --tissue --neighbours --json $MB/stage2/wc3_neighbours.json; log "pooled-rc=$?"

log "=== D2: paarweise nb vs. ohne nb ==="
$PY -m cmb.stage2.compare_runs --a $MB/stage2/wc3_neighbours.json "pooled+tissue+nb boosting" --b $MB/stage2/wc3_neighbours.json "pooled+tissue boosting" --cohorts valdo t2star swi; log "compare rc=$?"
log "=== TEIL D fertig ==="
