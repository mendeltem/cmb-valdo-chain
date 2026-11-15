#!/usr/bin/env bash
# wc7 Schritt 2+4: blob-Kandidaten (z40, wie der Pool S1-S3 + F10) und Vereinigungen (CPU).
# Reihenfolge: eine Kette, nacheinander (Schwere-Laeufe-Regel).
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

echo "=== [1/6] candidates VALDO vblob (z40) ==="
timeout 4200 $PY -m cmb.stage2.candidates \
  --predictions ergebnisse/turnier/a03-aniso-e60-gd-vblob \
  --grid dev/gitter_d --keep-csf --half 20 20 20 \
  --out ergebnisse/stage2/a03-aniso-e60-gd-vblob-weich-z40 || { echo "FEHLER valdo vblob"; exit 1; }

echo "=== [2/6] candidates T2* vblob (z40) ==="
timeout 4200 $PY -m cmb.stage2.candidates \
  --predictions "$MB/transfer/a03-aniso-e60-vblob-cat-t2s" \
  --manifest "$MB/gitter/manifest_t2star.json" \
  --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" \
  --keep-csf --half 20 20 20 \
  --out "$MB/stage2/a03-aniso-e60-vblob-cat-t2s-weich-z40" || { echo "FEHLER t2star vblob"; exit 1; }

echo "=== [3/6] candidates SWI vblob (z40) ==="
timeout 4200 $PY -m cmb.stage2.candidates \
  --predictions "$MB/transfer/a03-aniso-e60-vblob-cat-swi" \
  --manifest "$MB/gitter/manifest_swi.json" \
  --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" \
  --keep-csf --half 20 20 20 \
  --out "$MB/stage2/a03-aniso-e60-vblob-cat-swi-weich-z40" || { echo "FEHLER swi vblob"; exit 1; }

echo "=== [4/6] union VALDO (base + vblob) ==="
timeout 1800 $PY -m cmb.stage2.merge_candidates \
  --input ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 \
          ergebnisse/stage2/a03-aniso-e60-gd-vblob-weich-z40 \
  --min-runs 1 --out ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-union || { echo "FEHLER union valdo"; exit 1; }

echo "=== [5/6] union T2* (base + vblob) ==="
timeout 1800 $PY -m cmb.stage2.merge_candidates \
  --input "$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40" \
          "$MB/stage2/a03-aniso-e60-vblob-cat-t2s-weich-z40" \
  --min-runs 1 --out "$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40-union" || { echo "FEHLER union t2star"; exit 1; }

echo "=== [6/6] union SWI (base + vblob) ==="
timeout 1800 $PY -m cmb.stage2.merge_candidates \
  --input "$MB/stage2/a03-aniso-e60-cat-swi-weich-z40" \
          "$MB/stage2/a03-aniso-e60-vblob-cat-swi-weich-z40" \
  --min-runs 1 --out "$MB/stage2/a03-aniso-e60-cat-swi-weich-z40-union" || { echo "FEHLER union swi"; exit 1; }

echo "=== wc7 KANDIDATEN+UNION ALLE FERTIG ==="
