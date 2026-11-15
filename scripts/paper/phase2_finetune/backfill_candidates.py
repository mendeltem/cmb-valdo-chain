"""Refill candidate sets that were written empty because run_block passed pred/fold<k> instead of pred/ (fixed 01.11.2026).
Only blocks whose candidates.json says empty=True and whose prediction maps exist. CPU only. Prints aggregates only.
    python backfill_candidates.py
"""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import glob, json, os, subprocess
PAPER = _P + "/paper/phase4"
GRID = {"t2star": _P + "/gitter/manifest_t2star.json",
        "swi": _P + "/gitter/manifest_swi_v2.json"}
done = refilled = still_empty = 0
for cj in sorted(glob.glob(f"{PAPER}/candidates/*/*/candidates.json")):
    d = json.load(open(cj)); mod, block = cj.split("/")[-3], cj.split("/")[-2]
    if not d.get("empty"):
        done += 1; continue
    pred = f"{PAPER}/unet/{mod}/{block}/pred"
    if not glob.glob(f"{pred}/fold*/*_pred_proba.nii.gz"):
        continue
    out = os.path.dirname(cj)
    r = subprocess.run([_PY, "-m", "cmb.stage2.candidates", "--predictions", pred,
                        "--manifest", GRID[mod], "--out", out, "--threshold", "0.15", "--min-mm3", "1.0", "--half", "20", "20", "20",
                        "--keep-csf", "--csf-mode", "majority", "--synthseg", "none"],
                       cwd=_R, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if r.returncode == 0:
        refilled += 1
    else:
        still_empty += 1
        json.dump(d, open(cj, "w"), indent=1)          # keep the empty marker if it really has no candidate
    print(f"{mod} {block}: rc {r.returncode}", flush=True)
print(f"refilled {refilled}, still empty {still_empty}, already non-empty {done}")
