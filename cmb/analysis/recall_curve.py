"""Stage-1 recall ceiling vs candidates per scan (Arena 2, 2026-11-04). Aggregates only.

Same candidate logic as cmb.stage2.candidates (out-of-fold probability map, 26-connected components above a threshold,
min size in mm^3 at 0.25 mm^3 per voxel, components above MAX_COMPONENT_VOXELS dropped, CSF candidates KEPT as in the
frozen sets), swept over thresholds and minimum sizes; plus a second candidate rule: the top-K local maxima of the
probability map per scan (a peak 'touches' a reference lesion if the lesion has a voxel in its 3x3x3 neighbourhood).
For the reference lesions the default rule (0.15, 1 mm^3) misses: maximum stage-1 probability inside the lesion and size.

    python -m cmb.analysis.recall_curve --out RESULT.json \
        --cohort valdo=PRED_DIR:grid=GRID_DIR t2star=PRED_DIR:manifest=MANIFEST swi=PRED_DIR:manifest=MANIFEST
The default point (0.15, 1.0) must equal the stored ceiling recall of the frozen candidate sets (check printed).
"""
import argparse, glob, json, os
import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.stage2.candidates import MAX_COMPONENT_VOXELS

C26 = np.ones((3, 3, 3), bool)
THRESHOLDS = [0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
MIN_MM3 = [1.0, 0.25]
TOP_K = [10, 20, 30, 50, 100, 200, 400]


def cases_of(spec: str):
    pred, src = spec.split(":", 1)
    kind, where = src.split("=", 1)
    folders = None
    if kind == "manifest":
        folders = {e["id"]: e["dir"] for e in json.load(open(where))}
    for path in sorted(glob.glob(f"{pred}/fold*/*_pred_proba.nii.gz")):
        case = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        if folders is not None and case not in folders:
            continue
        folder = folders[case] if folders is not None else f"{where}/{case}"
        yield case, path, f"{folder}/{case}_label.nii.gz"


def analyse(spec: str) -> dict:
    comp = {(t, m): [0, 0] for t in THRESHOLDS for m in MIN_MM3}       # (found lesions, candidates)
    peaks = {k: [0, 0] for k in TOP_K}
    n_ref = n_cases = 0; missed_p, missed_v = [], []
    for case, ppath, lpath in cases_of(spec):
        prob = np.asarray(nib.load(ppath).dataobj).astype(np.float32) / 255.0
        ref, nr = ndimage.label(np.asarray(nib.load(lpath).dataobj) > 0, structure=C26)
        n_ref += nr; n_cases += 1
        for t in THRESHOLDS:
            lab, n = ndimage.label(prob > t, structure=C26)
            sizes = ndimage.sum(np.ones_like(prob), lab, range(1, n + 1)) if n else np.array([])
            for m in MIN_MM3:
                keep = np.zeros(n + 1, bool); keep[1:] = (sizes * 0.25 >= m) & (sizes <= MAX_COMPONENT_VOXELS)
                found = np.unique(ref[keep[lab] & (ref > 0)])
                comp[(t, m)][0] += len(found); comp[(t, m)][1] += int(keep.sum())
                if t == 0.15 and m == 1.0:
                    for les in set(range(1, nr + 1)) - set(found.tolist()):
                        inside = ref == les
                        missed_p.append(float(prob[inside].max())); missed_v.append(int(inside.sum()))
        loc = (prob == ndimage.maximum_filter(prob, size=3)) & (prob > 0.002)
        coords = np.argwhere(loc); order = np.argsort(-prob[loc])
        near = ndimage.maximum_filter(ref, size=3)                    # lesion id within the 3x3x3 neighbourhood
        for k in TOP_K:
            top = coords[order[:k]]
            hit = {int(near[tuple(c)]) for c in top} - {0}
            peaks[k][0] += len(hit); peaks[k][1] += len(top)
    r = lambda f, c: dict(recall=round(f / max(n_ref, 1), 4), candidates_per_scan=round(c / max(n_cases, 1), 2))
    mp = np.array(missed_p); mv = np.array(missed_v)
    return dict(n_cases=n_cases, n_reference=n_ref,
                components={f"t{t}_min{m}": r(*comp[(t, m)]) for t in THRESHOLDS for m in MIN_MM3},
                top_k_peaks={str(k): r(*peaks[k]) for k in TOP_K},
                missed_at_default=dict(n=len(mp),
                                       max_probability={b: int(((mp >= lo) & (mp < hi)).sum()) for b, lo, hi in
                                                        [("<0.01", 0, .01), ("0.01-0.05", .01, .05), ("0.05-0.10", .05, .10), ("0.10-0.15", .10, .15), (">=0.15 (too small/too big)", .15, 2)]},
                                       voxels={b: int(((mv >= lo) & (mv < hi)).sum()) for b, lo, hi in [("1-3", 1, 4), ("4-9", 4, 10), ("10-29", 10, 30), (">=30", 30, 10**9)]}))


ap = argparse.ArgumentParser()
ap.add_argument("--cohort", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {}
for c in a.cohort:
    name, spec = c.split("=", 1)
    res[name] = analyse(spec)
    d = res[name]["components"]["t0.15_min1.0"]
    print(f"{name}: default rule recall {d['recall']} at {d['candidates_per_scan']} candidates/scan "
          f"({res[name]['n_cases']} cases, {res[name]['n_reference']} lesions)", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
