"""Arena 2 stage-1 arms vs the frozen control (2026-11-05). Aggregates only.

Same candidate rule as cmb.stage2.candidates (out-of-fold probability > threshold, 26-connected components, at least
MIN_MM3 at 0.25 mm^3 per voxel, components above MAX_COMPONENT_VOXELS dropped) and the same size classes as
cmb.analysis.size_stratified (diameter of the circle with the area of the largest cross-section on one slice).
Per cohort and model: reference CMB per size class, stage-1 ceiling recall (touched by any candidate), candidates per
scan; per arm the paired difference to the control at lesion level (found by arm only / control only, exact McNemar p).

    python -m cmb.analysis.s1_arm_compare --out RESULT.json --control DIR --arms NAME=DIR ... --manifest MANIFEST
"""
import argparse, glob, json, os
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import binomtest

from cmb.stage2.candidates import MAX_COMPONENT_VOXELS

C26 = np.ones((3, 3, 3), bool)
CLASSES = (("<2 mm", 0, 2), ("2-5 mm", 2, 5), (">5 mm", 5, 1e9))
THRESHOLDS = (0.15, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005)   # low ones: compare at matched candidates per scan
MIN_MM3 = 1.0


def size_class(d):
    return next(name for name, lo, hi in CLASSES if lo <= d < hi or (name == "2-5 mm" and d == 5))


def reference(label_path):
    img = nib.load(label_path); zooms = img.header.get_zooms()[:3]
    ref, n = ndimage.label(np.asarray(img.dataobj) > 0, structure=C26)
    ax = int(np.argmax(zooms)); inplane = np.prod([z for i, z in enumerate(zooms) if i != ax])
    cls = {}
    for les, box in enumerate(ndimage.find_objects(ref), 1):
        area = (ref[box] == les).sum(axis=tuple(i for i in range(3) if i != ax)).max() * inplane
        cls[les] = size_class(float(2 * np.sqrt(area / np.pi)))
    return ref, cls


def found(prob, ref, t):
    lab, n = ndimage.label(prob > t, structure=C26)
    if not n:
        return set(), 0
    sizes = ndimage.sum(np.ones_like(prob), lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = (sizes * 0.25 >= MIN_MM3) & (sizes <= MAX_COMPONENT_VOXELS)
    return set(np.unique(ref[keep[lab] & (ref > 0)]).tolist()) - {0}, int(keep.sum())


def probas(d):
    return {os.path.basename(p)[:-len("_pred_proba.nii.gz")]: p for p in glob.glob(f"{d}/fold*/*_pred_proba.nii.gz")}


ap = argparse.ArgumentParser()
ap.add_argument("--control", required=True); ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
folders = {e["id"]: e["dir"] for e in json.load(open(a.manifest))}
models = {"control": probas(a.control), **{s.split("=", 1)[0]: probas(s.split("=", 1)[1]) for s in a.arms}}
cases = sorted(set.intersection(*(set(m) for m in models.values())) & set(folders))
assert all(len(m) == len(cases) for m in models.values()), {k: len(v) for k, v in models.items()}

hits = {m: {t: [] for t in THRESHOLDS} for m in models}      # per lesion: (class, found)
ncand = {m: {t: 0 for t in THRESHOLDS} for m in models}
for case in cases:
    ref, cls = reference(f"{folders[case]}/{case}_label.nii.gz")
    for m, paths in models.items():
        prob = np.asarray(nib.load(paths[case]).dataobj).astype(np.float32) / 255.0
        for t in THRESHOLDS:
            f, k = found(prob, ref, t); ncand[m][t] += k
            hits[m][t] += [(cls[les], les in f) for les in sorted(cls)]

res = dict(n_cases=len(cases), definition="ceiling recall = reference CMB touched by a stage-1 candidate "
           f"(prob > t, >= {MIN_MM3} mm^3); size class by the largest axial cross-section", models={})
for m in models:
    res["models"][m] = {}
    for t in THRESHOLDS:
        h = hits[m][t]; row = dict(candidates_per_scan=round(ncand[m][t] / len(cases), 2))
        for cname in [c[0] for c in CLASSES] + ["all"]:
            sel = [ok for c, ok in h if cname == "all" or c == cname]
            row[cname] = dict(n=len(sel), recall=round(sum(sel) / max(len(sel), 1), 3))
        if m != "control":
            ctl = hits["control"][t]
            only_arm = sum(x[1] and not y[1] for x, y in zip(h, ctl)); only_ctl = sum(y[1] and not x[1] for x, y in zip(h, ctl))
            row["vs_control"] = dict(found_by_arm_only=only_arm, found_by_control_only=only_ctl,
                                     mcnemar_p=round(binomtest(only_arm, only_arm + only_ctl).pvalue, 4) if only_arm + only_ctl else 1.0)
        res["models"][m][f"t{t}"] = row
json.dump(res, open(a.out, "w"), indent=1)
for m, r in res["models"].items():
    d = r["t0.15"]
    print(f"{m:10s} t0.15: all {d['all']['recall']} (<2 {d['<2 mm']['recall']}, 2-5 {d['2-5 mm']['recall']}, >5 {d['>5 mm']['recall']}), "
          f"{d['candidates_per_scan']} cand/scan {d.get('vs_control', '')}", flush=True)
