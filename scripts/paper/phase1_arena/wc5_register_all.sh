#!/bin/bash
# wc5: register venous atlas auf alle Faelle, 4 Kohorten nacheinander (CPU, 12 Threads).
# Wiederaufnehmbar: vorhandene <case>_vein.nii.gz werden uebersprungen.
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -o pipefail
PY=${CMB_PYTHON:-python}
MB=$CMB_PRIVATE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd $CMB_ROOT || exit 1
echo "== valdo $(date)"
nice -n 10 $PY -m cmb.stage2.atlas_features register --grid dev/gitter_d \
  --cases-from ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 --out dev/atlas_vein --threads 12
[ $? -eq 0 ] || exit 1
echo "== t2star $(date)"
nice -n 10 $PY -m cmb.stage2.atlas_features register --manifest $MB/gitter/manifest_t2star.json \
  --cases-from $MB/stage2/a03-aniso-e60-cat-t2s-weich-z40 --out $MB/atlas_vein/t2star --threads 12
[ $? -eq 0 ] || exit 1
echo "== swi $(date)"
nice -n 10 $PY -m cmb.stage2.atlas_features register --manifest $MB/gitter/manifest_swi.json \
  --cases-from $MB/stage2/a03-aniso-e60-cat-swi-weich-z40 --out $MB/atlas_vein/swi --threads 12
[ $? -eq 0 ] || exit 1
echo "== momeni (extern, Bestaetigung spaeter) $(date)"
nice -n 10 $PY -m cmb.stage2.atlas_features register --manifest ${CMB_MOMENI:-$CMB_ROOT/data/momeni-csiro-cmb}/gitter/manifest_momeni.json \
  --cases-from ${CMB_MOMENI:-$CMB_ROOT/data/momeni-csiro-cmb}/stage2/zero-shot-swi-s42-weich --out ${CMB_MOMENI:-$CMB_ROOT/data/momeni-csiro-cmb}/atlas_vein --threads 12
[ $? -eq 0 ] || exit 1
echo "== ALLE REGISTER FERTIG $(date)"
