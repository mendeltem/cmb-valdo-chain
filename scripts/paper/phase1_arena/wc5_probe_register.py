"""wc5: Probelauf register mit 2 Faellen (valdo) + Zeitmessung; zaehlt fehlende Bilder je Kohorte."""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json
import os
import subprocess
import sys
import time

ROOT = _R
PY = _PY
MB = _P
COHORTS = [
    ("valdo", "dev/gitter_d", [], "ergebnisse/stage2/a03-aniso-e60-gd-weich-z40"),
    ("t2star", None, [f"{MB}/gitter/manifest_t2star.json"], f"{MB}/stage2/a03-aniso-e60-cat-t2s-weich-z40"),
    ("swi", None, [f"{MB}/gitter/manifest_swi.json"], f"{MB}/stage2/a03-aniso-e60-cat-swi-weich-z40"),
]

# fehlende Bilder? (register wuerde mit check=True abcrashen)
for name, grid, manifests, cand in COHORTS:
    p = cand if cand.endswith(".json") else f"{cand if os.path.isabs(cand) else f'{ROOT}/{cand}'}/candidates.json"
    cases = json.load(open(p))["cases"]
    if manifests:
        dirs = {e["id"]: e["dir"] for m in manifests for e in json.load(open(m))}
        img = lambda c: f"{dirs[c]}/{c}_image.nii.gz"
    else:
        img = lambda c: f"{ROOT}/{grid}/{c}/{c}_image.nii.gz"
    noimg = [c for c in cases if not os.path.exists(img(c))]
    print(f"{name}: {len(cases)} cases, Bild fehlt: {len(noimg)} {noimg[:3]}")

# Probelauf: 2 Faelle valdo
tmp = f"{ROOT}/skripte/wc5_probe"
os.makedirs(tmp, exist_ok=True)
info = json.load(open(f"{ROOT}/ergebnisse/stage2/a03-aniso-e60-gd-weich-z40/candidates.json"))
json.dump({"cases": {c: info["cases"][c] for c in list(info["cases"])[:2]}}, open(f"{tmp}/candidates.json", "w"))
cmd = [PY, "-m", "cmb.stage2.atlas_features", "register", "--grid", "dev/gitter_d",
       "--cases-from", "skripte/wc5_probe", "--out", "dev/atlas_vein_probe", "--threads", "12"]
t0 = time.time()
subprocess.run(cmd, cwd=ROOT, check=True)
print(f"2 cases in {time.time() - t0:.0f} s -> ~{(time.time() - t0) / 2 / 60:.2f} min/case")
