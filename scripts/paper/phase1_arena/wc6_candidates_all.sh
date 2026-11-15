#!/usr/bin/env bash
# wc6 Schritt 5: Candidates der synthetischen Kopien (CPU), wie der Pool (S1-S3 + F10).
# --keep-csf --half 20 20 20, exakt aus dem Auftrag.
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

echo "=== candidates valdo-s0 (57) ==="
timeout 3000 $PY -m cmb.stage2.candidates \
  --predictions ergebnisse/synthetic/valdo-s0/maps \
  --manifest ergebnisse/synthetic/valdo-s0/manifest_synthetic.json \
  --synthseg "dev/synthseg/{id}_synthseg.nii.gz" \
  --keep-csf --half 20 20 20 \
  --out ergebnisse/synthetic/valdo-s0/candidates || { echo "FEHLER valdo-s0"; exit 1; }

echo "=== candidates ${CMB_LABEL}-t2star-s0 (75) ==="
timeout 3000 $PY -m cmb.stage2.candidates \
  --predictions "$MB/synthetic/${CMB_LABEL}-t2star-s0/maps" \
  --manifest "$MB/synthetic/${CMB_LABEL}-t2star-s0/manifest_synthetic.json" \
  --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" \
  --keep-csf --half 20 20 20 \
  --out "$MB/synthetic/${CMB_LABEL}-t2star-s0/candidates" || { echo "FEHLER t2star"; exit 1; }

echo "=== candidates ${CMB_LABEL}-swi-s0 (61) ==="
timeout 3000 $PY -m cmb.stage2.candidates \
  --predictions "$MB/synthetic/${CMB_LABEL}-swi-s0/maps" \
  --manifest "$MB/synthetic/${CMB_LABEL}-swi-s0/manifest_synthetic.json" \
  --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" \
  --keep-csf --half 20 20 20 \
  --out "$MB/synthetic/${CMB_LABEL}-swi-s0/candidates" || { echo "FEHLER swi"; exit 1; }

echo "=== wc6 candidates ALLE FERTIG ==="
