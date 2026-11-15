#!/bin/sh
# wk7: Zusatz-Worker (8 x 2 Threads) fuer die NOCH fehlenden Prognosen.
# Liest /tmp/wk7_jobs.txt, springt vorhandene Karten ueber (Worker ist wiederaufnehmbar).
# POSIX-sh, fuer den hintergrund-Runner.
set -u
cd "${CMB_ROOT:-$(dirname "$(readlink -f "$0")")/..}"
PY=${CMB_PYTHON:-python}
OUT=ergebnisse/schichtdicke/proba
$PY - <<'EOF' > /tmp/wk7_jobs_rest.txt
import os, json
man = {}
for line in open("/tmp/wk7_jobs.txt"):
    name, dicke, m, sid, mpfad, out = line.split()
    man.setdefault(m, json.load(open(m)))
for line in open("/tmp/wk7_jobs.txt"):
    name, dicke, m, sid, mpfad, out = line.split()
    k = next(e["fold"] for e in man[m] if e["id"] == sid)
    p = f"ergebnisse/schichtdicke/proba/{name}/{dicke}/fold{k}/{sid}_pred_proba.nii.gz"
    if not os.path.exists(p):
        print(line.rstrip())
EOF
wc -l /tmp/wk7_jobs_rest.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
xargs -P 6 -n 6 $PY code/wk7_predict_worker.py < /tmp/wk7_jobs_rest.txt > /tmp/wk7_predict_more.log 2>&1
echo "zusatz fertig, rc=$?"
find $OUT -name '*pred_proba*' | wc -l
