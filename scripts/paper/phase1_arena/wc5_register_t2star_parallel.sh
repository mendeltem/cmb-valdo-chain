#!/bin/bash
# wc5: t2star-Register PARALLEL zum wc5-register valdo-Block (24 Kerne, 12+12 Threads).
# Wiederaufnehmbar: vorhandene <case>_vein.nii.gz werden uebersprungen.
# Wenn der valdo-Block im Hauptskript anschliesst, ueberspringt er hier alle Faelle.
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
echo "== t2star (parallel) $(date)"
nice -n 10 $PY -m cmb.stage2.atlas_features register --manifest $MB/gitter/manifest_t2star.json \
  --cases-from $MB/stage2/a03-aniso-e60-cat-t2s-weich-z40 --out $MB/atlas_vein/t2star --threads 12
[ $? -eq 0 ] || exit 1
echo "== t2star FERTIG $(date)"
