#!/usr/bin/env bash
# wc6 Schritt 3: Insert (CPU, seed 0, 3 Blutungen + 3 Venen), eine Kette je (sub-)Kohorte.
# VALDO: 3 Sub-Kohorten -> ergebnisse/synthetic/valdo-<sub>-s0, danach merge nach valdo-s0/.
# Private: $MB/synthetic/<cohort>-s0.
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
mkdir -p ergebnisse/synthetic "$MB/synthetic"

echo "=== insert valdo-1 (SABRE: TE 19.9, slice 4.0, inplane 0.45) ==="
$PY -m cmb.synthesis.insert_volume --manifest dev/manifests_wc6/manifest_valdo_1_gitter_d.json \
  --synthseg "dev/synthseg/{id}_synthseg.nii.gz" \
  --b0 3 --te 19.9 --slice-mm 4.0 --inplane-mm 0.45 \
  --bleeds 3 --veins 3 --seed 0 --out ergebnisse/synthetic/valdo-1-s0 || { echo "FEHLER valdo-1"; exit 1; }

echo "=== insert valdo-2 (Rotterdam: TE 30.0, slice 0.8, inplane 0.49) ==="
$PY -m cmb.synthesis.insert_volume --manifest dev/manifests_wc6/manifest_valdo_2_gitter_d.json \
  --synthseg "dev/synthseg/{id}_synthseg.nii.gz" \
  --b0 3 --te 30.0 --slice-mm 0.8 --inplane-mm 0.49 \
  --bleeds 3 --veins 3 --seed 0 --out ergebnisse/synthetic/valdo-2-s0 || { echo "FEHLER valdo-2"; exit 1; }

echo "=== insert valdo-3 (ALFA: TE 29.0, slice 3.0, inplane 1.0) ==="
$PY -m cmb.synthesis.insert_volume --manifest dev/manifests_wc6/manifest_valdo_3_gitter_d.json \
  --synthseg "dev/synthseg/{id}_synthseg.nii.gz" \
  --b0 3 --te 29.0 --slice-mm 3.0 --inplane-mm 1.0 \
  --bleeds 3 --veins 3 --seed 0 --out ergebnisse/synthetic/valdo-3-s0 || { echo "FEHLER valdo-3"; exit 1; }

echo "=== merge valdo-s0 (Manifeste concat, Logs merge) ==="
$PY - <<'EOF'
import json, os
V = __import__("os").environ["CMB_ROOT"]
out = f"{V}/ergebnisse/synthetic/valdo-s0"
os.makedirs(out, exist_ok=True)
man = []
log = {}
for sub in (1, 2, 3):
    d = f"{V}/ergebnisse/synthetic/valdo-{sub}-s0"
    man += json.load(open(f"{d}/manifest_synthetic.json"))
    lg = json.load(open(f"{d}/insert_log.json"))
    if isinstance(lg, dict):
        for k, v in lg.items():
            if k in log and isinstance(v, list):
                log[k] += v
            elif k in log and isinstance(v, dict):
                log[k].update(v)
            else:
                log[k] = v
    else:
        log["entries"] = log.get("entries", []) + lg
json.dump(man, open(f"{out}/manifest_synthetic.json", "w"), indent=1)
json.dump(log, open(f"{out}/insert_log.json", "w"), indent=1)
print("merged", len(man), "cases ->", out)
EOF
[ $? -eq 0 ] || { echo "FEHLER merge"; exit 1; }

echo "=== insert ${CMB_LABEL}-t2star (TE 19.9, slice 5.0, inplane 0.69) ==="
$PY -m cmb.synthesis.insert_volume --manifest "$MB/gitter/manifest_t2star.json" \
  --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" \
  --b0 3 --te 19.9 --slice-mm 5.0 --inplane-mm 0.69 \
  --bleeds 3 --veins 3 --seed 0 --out "$MB/synthetic/${CMB_LABEL}-t2star-s0" || { echo "FEHLER t2star"; exit 1; }

echo "=== insert ${CMB_LABEL}-swi (TE 20.0, slice 2.0, inplane 0.57) ==="
$PY -m cmb.synthesis.insert_volume --manifest "$MB/gitter/manifest_swi.json" \
  --synthseg "$MB/synthseg-swi/robust/{sid}_synthseg.nii.gz" \
  --b0 3 --te 20.0 --slice-mm 2.0 --inplane-mm 0.57 \
  --bleeds 3 --veins 3 --seed 0 --out "$MB/synthetic/${CMB_LABEL}-swi-s0" || { echo "FEHLER swi"; exit 1; }

echo "=== wc6 insert ALLE FERTIG ==="
