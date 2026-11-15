#!/bin/bash
# wc5: TEIL-AUC fuer valdo mit den BEREITS fertigen Register-Maps (Sofort-Nachweis, R3).
# Nur die Faelle mit fertiger Map (SCRATCH-Ordner, NICHT die -atlas-Ordner -- die
# schreibt wc5-rest erst nach komplettem Register mit ALLEN Faellem ueber).
: "${CMB_ROOT:=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}"   # paper version: paths from the environment
: "${CMB_PRIVATE:=$CMB_ROOT/private}"; : "${CMB_LOGS:=$CMB_PRIVATE/logs}"; : "${CMB_LABEL:=${CMB_INHOUSE_LABEL:-inhouse}}"; export CMB_ROOT CMB_PRIVATE
# optional GPU queue (CMB_QUEUE, e.g. a scheduler wrapper taking "... -- cmd"); without it the command runs directly
queue() { if [ -n "${CMB_QUEUE:-}" ]; then $CMB_QUEUE "$@"; else while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; "$@"; fi; }
# optional wait for a named background job (CMB_WAIT); without it: no wait
waitjob() { if [ -n "${CMB_WAIT:-}" ]; then $CMB_WAIT "$@"; else return 0; fi; }
set -u
PY=${CMB_PYTHON:-python}
V=$CMB_ROOT
RES=$V/ergebnisse/stage2/wc5_rest
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$V" || exit 1
mkdir -p "$RES"
SRC="$RES/a03-gd-TEIL-valdo"
rm -rf "$SRC" "$SRC-atlas"
"$PY" - "$SRC" << 'EOF'
import json, os, sys
dst = sys.argv[1]
base = __import__("os").environ["CMB_ROOT"]
info = json.load(open(f"{base}/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40/candidates.json"))
have = {f[:-len("_vein.nii.gz")] for f in os.listdir(f"{base}/dev/atlas_vein")
        if f.endswith("_vein.nii.gz") and not f.startswith("atlas_on_template")}
rows = [r for r in info["candidates"] if r["case"] in have]
out = {k: info[k] for k in info if k != "candidates"}
out["candidates"] = rows
out["cases"] = {c: v for c, v in info["cases"].items() if c in have}
os.makedirs(dst, exist_ok=True)
json.dump(out, open(f"{dst}/candidates.json", "w"))
print(f"Teilmengen: {len(have)} Faelle mit Map, {len(rows)} Kandidaten")
EOF
echo "subset rc=$?"
nice -n 10 "$PY" -m cmb.stage2.atlas_features features \
  --candidates "$SRC" --maps "dev/atlas_vein/{id}_vein.nii.gz" \
  --out "$SRC-atlas" >> "$RES/teilvaldo.log" 2>&1
echo "features rc=$?"
n=$(ls dev/atlas_vein/*_vein.nii.gz | grep -cv atlas_on_template)
nice -n 10 "$PY" -m cmb.stage2.atlas_features auc \
  --candidates "valdo-teil($n/57)=$SRC-atlas" > "$RES/auc_teilvaldo.txt" 2>&1
echo "auc rc=$?"
cat "$RES/auc_teilvaldo.txt"
