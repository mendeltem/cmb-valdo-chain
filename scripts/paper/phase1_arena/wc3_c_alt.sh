#!/bin/bash
# wc3 Teil C: unabhaengige Kandidatenquelle (frst999, frst995, gmm) auf allen drei Kohorten,
# Ceilings, dann gepoolt + Vergleich gegen Basis. LAUFT IM VORDERGRUND (Teil A/D laufen parallel).
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

log "=== C1: alt-Kandidaten VALDO (frst999, frst995, gmm) ==="
$PY -m cmb.stage2.alt_candidates --base ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --mode frst --out ergebnisse/turnier/wc3-alt-frst999; log "alt valdo frst999 rc=$?"
$PY -m cmb.stage2.alt_candidates --base ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --mode frst --frst-quantile 99.5 --out ergebnisse/turnier/wc3-alt-frst995; log "alt valdo frst995 rc=$?"
$PY -m cmb.stage2.alt_candidates --base ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --mode gmm --out ergebnisse/turnier/wc3-alt-gmm; log "alt valdo gmm rc=$?"

log "=== C2: Ceiling VALDO (base + 3 Varianten) ==="
$PY -m cmb.analysis.candidate_ceiling --run base=ergebnisse/turnier/a03-aniso-e60-gd frst999=ergebnisse/turnier/wc3-alt-frst999 frst995=ergebnisse/turnier/wc3-alt-frst995 gmm=ergebnisse/turnier/wc3-alt-gmm --grid dev/gitter_d --thresholds 0.15 --json ergebnisse/analysis/wc3_ceiling_alt.json; log "ceiling-alt rc=$?"

log "=== C3: alt-Kandidaten T2* ==="
$PY -m cmb.stage2.alt_candidates --base $MB/transfer/a03-aniso-e60-cat-t2s --manifest $MB/gitter/manifest_t2star.json --mode frst --out $MB/transfer/wc3-alt-frst999-cat-t2s; log "alt t2s frst999 rc=$?"
$PY -m cmb.stage2.alt_candidates --base $MB/transfer/a03-aniso-e60-cat-t2s --manifest $MB/gitter/manifest_t2star.json --mode frst --frst-quantile 99.5 --out $MB/transfer/wc3-alt-frst995-cat-t2s; log "alt t2s frst995 rc=$?"
$PY -m cmb.stage2.alt_candidates --base $MB/transfer/a03-aniso-e60-cat-t2s --manifest $MB/gitter/manifest_t2star.json --mode gmm --out $MB/transfer/wc3-alt-gmm-cat-t2s; log "alt t2s gmm rc=$?"

log "=== C4: alt-Kandidaten SWI ==="
$PY -m cmb.stage2.alt_candidates --base $MB/transfer/a03-aniso-e60-cat-swi --manifest $MB/gitter/manifest_swi.json --mode frst --out $MB/transfer/wc3-alt-frst999-cat-swi; log "alt swi frst999 rc=$?"
$PY -m cmb.stage2.alt_candidates --base $MB/transfer/a03-aniso-e60-cat-swi --manifest $MB/gitter/manifest_swi.json --mode frst --frst-quantile 99.5 --out $MB/transfer/wc3-alt-frst995-cat-swi; log "alt swi frst995 rc=$?"
$PY -m cmb.stage2.alt_candidates --base $MB/transfer/a03-aniso-e60-cat-swi --manifest $MB/gitter/manifest_swi.json --mode gmm --out $MB/transfer/wc3-alt-gmm-cat-swi; log "alt swi gmm rc=$?"

log "=== C5: Kandidaten (Ceiling) der alt-Varianten, privater Kohorten (REPLIKATION S2/S3-Muster) ==="
for VAR in frst999 frst995 gmm; do
  $PY -m cmb.stage2.candidates --predictions $MB/transfer/wc3-alt-$VAR-cat-t2s --manifest $MB/gitter/manifest_t2star.json --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/wc3-alt-$VAR-cat-t2s-weich; log "cand t2s $VAR rc=$?"
  $PY -m cmb.stage2.candidates --predictions $MB/transfer/wc3-alt-$VAR-cat-swi --manifest $MB/gitter/manifest_swi.json --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/wc3-alt-$VAR-cat-swi-weich; log "cand swi $VAR rc=$?"
done
log "=== TEIL C1-C5 fertig ==="
