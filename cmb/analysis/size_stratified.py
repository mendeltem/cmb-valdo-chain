"""CMB results by size class: < 2 mm, 2-5 mm, > 5 mm (user rule 2026-11-05). Aggregates only.

Diameter of a reference CMB = diameter of the circle with the area of its LARGEST cross-section on one slice (slice axis =
the axis with the largest voxel spacing), the same definition as the rater tool. Reference lesions are 26-connected
components, numbered exactly as in cmb.stage2.candidates (so the 'touched' ids of the candidates match).

Input: a decision dump of cmb.stage2.pooled (--dump-decisions FILE; one JSON line per mode and candidate) and the
label files of the cohorts. Per cohort and class: number of reference CMB, stage-1 ceiling recall (touched by ANY
candidate) and chain recall per stage-2 mode (touched by an ACCEPTED candidate).

    python -m cmb.analysis.size_stratified --decisions DUMP.jsonl --out RESULT.json \
        --labels valdo=grid:GRID_DIR t2star=manifest:MANIFEST swi=manifest:MANIFEST [--modes "pooled feat-vessel-tissue-atlas"]
"""
import argparse, json
from collections import defaultdict
import nibabel as nib
import numpy as np
from scipy import ndimage

C26 = np.ones((3, 3, 3), bool)
CLASSES = (("<2 mm", 0, 2), ("2-5 mm", 2, 5), (">5 mm", 5, 1e9))


def size_class(d: float) -> str:
    return next(name for name, lo, hi in CLASSES if lo <= d < hi or (name == "2-5 mm" and d == 5))


def diameters(label_path: str) -> dict:
    img = nib.load(label_path); zooms = img.header.get_zooms()[:3]
    ref, n = ndimage.label(np.asarray(img.dataobj) > 0, structure=C26)
    ax = int(np.argmax(zooms)); inplane = np.prod([z for i, z in enumerate(zooms) if i != ax])
    out = {}
    for les, box in enumerate(ndimage.find_objects(ref), 1):
        m = ref[box] == les
        area = m.sum(axis=tuple(i for i in range(3) if i != ax)).max() * inplane
        out[les] = float(2 * np.sqrt(area / np.pi))
    return out


def label_files(spec: str) -> dict:
    kind, where = spec.split(":", 1)
    if kind == "manifest":
        return {e["id"]: f"{e['dir']}/{e['id']}_label.nii.gz" for e in json.load(open(where))}
    import os
    return {c: f"{where}/{c}/{c}_label.nii.gz" for c in sorted(os.listdir(where)) if os.path.isdir(f"{where}/{c}")}


ap = argparse.ArgumentParser()
ap.add_argument("--decisions", required=True); ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--modes", nargs="*", default=None); ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.decisions)]
modes = a.modes or sorted({r["mode"] for r in rows})
touched_any = defaultdict(set); touched_acc = defaultdict(set)
for r in rows:
    key = (r["cohort"], r["case"])
    touched_any[key].update(r["touched"])
    if r["accepted"]:
        touched_acc[(r["mode"],) + key].update(r["touched"])

res = {}
for spec in a.labels:
    cohort, src = spec.split("=", 1)
    files = label_files(src)
    cases = sorted({r["case"] for r in rows if r["cohort"] == cohort})
    count = defaultdict(int); ceil = defaultdict(int); chain = {m: defaultdict(int) for m in modes}; diam_all = []
    for case in cases:
        for les, d in diameters(files[case]).items():
            c = size_class(d); diam_all.append(d)
            for cls in (c, "all"):
                count[cls] += 1
                ceil[cls] += les in touched_any[(cohort, case)]
                for m in modes:
                    chain[m][cls] += les in touched_acc[(m, cohort, case)]
    res[cohort] = {cls: dict(n=count[cls], share=round(count[cls] / max(count["all"], 1), 3),
                             stage1_ceiling_recall=round(ceil[cls] / max(count[cls], 1), 3),
                             chain_recall={m: round(chain[m][cls] / max(count[cls], 1), 3) for m in modes})
                   for cls in [c[0] for c in CLASSES] + ["all"]}
    res[cohort]["median_diameter_mm"] = round(float(np.median(diam_all)), 2) if diam_all else None
    print(cohort, {cls: (v["n"], v["stage1_ceiling_recall"]) for cls, v in res[cohort].items() if isinstance(v, dict)}, flush=True)
json.dump(dict(definition="diameter = circle of the largest cross-section on one slice; classes <2, 2-5, >5 mm",
               decisions=a.decisions, modes=modes, cohorts=res), open(a.out, "w"), indent=1)
