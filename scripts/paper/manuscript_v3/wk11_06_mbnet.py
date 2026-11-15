#!/usr/bin/env python
"""wk11e step 5: microbleednet method check (REPORT ONLY, CPU only).

Reads only the two result summaries MB/extern/v2_mbnet_{cohort}.json and the header of one
preprocessed image per cohort -- no whole-JSON dump, no per-case lists, no volume data.
Writes $P/wk11_mbnetcheck.json. Nothing is changed.
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
import json, glob
import nibabel as nib
import numpy as np

E = _P + "/extern"
P = _R + "/paper"
out = {}

for coh in ["swi", "t2star"]:
    d = json.load(open(f"{E}/v2_mbnet_{coh}.json"))
    r = dict(n_cases=d["n_cases"], postprocessing=d["postprocessing"])
    for k in ["author_threshold", "nested"]:
        v = d[k]
        r[k] = dict(label=v["label"].split("(")[0].strip(), f1=v["f1"], sensitivity=v["sensitivity"],
                    precision=v["precision"], fp_per_case=v["fp_per_case"],
                    tp=v["tp"], fp=v["fp"], fn=v["fn"])
    for k in ["components_author", "components_nested"]:
        v = d[k]
        r[k] = dict(n_components=v["n_components"], n_kept=v["n_kept"],
                    n_dropped_small=v["n_dropped_small"], n_dropped_large=v["n_dropped_large"])
    # geometry of what the network actually saw (one file, header only)
    f = sorted(glob.glob(f"{E}/mbnet-{coh}-pre/images/*_preproc.nii.gz"))[0]
    ni = nib.load(f)
    z = np.asarray(ni.header.get_zooms()[:3], dtype=float)
    vv = float(np.prod(z))
    r["geometry"] = dict(zoom=[round(float(x), 3) for x in z], voxel_mm3=round(vv, 3),
                        n_preproc_files=len(glob.glob(f"{E}/mbnet-{coh}-pre/images/*_preproc.nii.gz")))
    # the arena minimum-size rule in mm3 and its sphere-diameter equivalent
    r["min_size_rule"] = dict(min_voxels=4, min_mm3=1.0,
                              min_mm3_at_this_grid=round(4 * vv, 2),
                              sphere_diameter_mm=round(2 * (3 * 4 * vv / (4 * np.pi)) ** (1 / 3), 2))
    r["share_dropped_small"] = round(r["components_author"]["n_dropped_small"] / r["components_author"]["n_components"], 3)
    out[coh] = r

json.dump(out, open(f"{P}/wk11_mbnetcheck.json", "w"), indent=1)
print(json.dumps(out, indent=1))
