#!/bin/sh
# wk7: restliche Prognose-Karten (Dicken n3/n5/n7) parallel auf der CPU.
# Liste: /tmp/wk7_ordered.txt (base-Modell zuerst = kritischer Pfad), 6 Argumente je Zeile.
# Worker ist wiederaufnehmbar (vorhandene Karte -> uebersprungen), daher beliebig oft neu startbar.
set -u
cd "${CMB_ROOT:-$(dirname "$(readlink -f "$0")")/..}"
PY=${CMB_PYTHON:-python}
OUT=ergebnisse/schichtdicke/proba
N_WORKER=${WK7_WORKER:-4}
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
echo "START $(date +%H:%M:%S) worker=$N_WORKER jobs=$(wc -l < /tmp/wk7_ordered.txt)"
xargs -P "$N_WORKER" -n 6 "$PY" -u code/wk7_predict_worker.py < /tmp/wk7_ordered.txt
echo "xargs rc=$? $(date +%H:%M:%S)"
echo "Karten: $(find $OUT -name '*pred_proba*' | wc -l)"
