"""Score one or more runs per cohort with ONE fixed post-processing, so strategies are comparable.

Every run stores out-of-fold probability maps. Here they all get the same treatment -- threshold
0.3, minimum size 2 mm3, giant components removed -- which was selected on VALDO with cross-fold
selection (2026-09-19) and is NOT re-tuned per run or per cohort: tuning it on the private cohort
would leak, and per-run tuning would reward luck. The CSF rule is reported separately and only
where a SynthSeg map exists.

Output: one line per run and cohort with pooled lesion F1 / precision / sensitivity / false
positives per case, the per-subject median F1 (cases with reference lesions), and the breakdown by
lesion-count class. Sums only -- safe to show for the private cohort.

    python -m cmb.transfer.evaluate --run NAME=FOLDER [NAME=FOLDER ...] --manifest M1.json [M2.json ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
from metric import hits, summarize            # noqa: E402

THRESHOLD, MIN_MM3, MAX_VOXELS, VOXEL_MM3 = 0.3, 2.0, 2100, 0.25
CONNECTIVITY_26 = np.ones((3, 3, 3), bool)


def count_class(n: int) -> str:
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else ">5"


def post_process(probability: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(probability > THRESHOLD, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= MIN_MM3) & (size <= MAX_VOXELS); keep[0] = False
    return keep[labels]


def f1_of(rows: List[Dict]) -> float:
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="NAME=FOLDER")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--json", help="write the table here as well")
    parser.add_argument("--split-valdo", action="store_true", help="report the three VALDO cohorts (valdo-1/2/3: 4 mm, 0.8 mm, 3-4 mm slices) separately")
    args = parser.parse_args()
    cases = {e["id"]: e for p in args.manifest for e in json.load(open(p))}
    table = {}
    per_case: Dict[str, Dict[str, Dict[str, Dict]]] = {}      # run -> cohort -> case id -> hits row (for paired comparisons)
    for spec in args.run:
        name, folder = spec.split("=", 1)
        folder = folder if os.path.isabs(folder) else f"{ROOT}/{folder}"
        per_cohort: Dict[str, List[Dict]] = {}
        for path in sorted(glob.glob(f"{folder}/fold*/*_pred_proba.nii.gz")):
            case_id = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
            if case_id not in cases:
                continue
            case = cases[case_id]
            probability = np.asarray(nib.load(path).dataobj).astype(np.float32) / 255.0
            reference = np.asarray(nib.load(f"{case['dir']}/{case_id}_label.nii.gz").dataobj) > 0
            row = hits(reference, post_process(probability)); row["class"] = count_class(row["n_ref"])
            cohort = "valdo" if case["cohort"].startswith("valdo") else case["cohort"]     # VALDO collapses to one group unless --split-valdo; any other cohort keeps its name
            if args.split_valdo and case["cohort"].startswith("valdo"):
                cohort = case["cohort"]
            per_cohort.setdefault(cohort, []).append(row)
            per_case.setdefault(name, {}).setdefault(cohort, {})[case_id] = row
        table[name] = {}
        for cohort, rows in sorted(per_cohort.items()):
            s = summarize(rows)
            with_lesions = [f1_of([r]) for r in rows if r["n_ref"] > 0]
            by_class = {c: round(f1_of([r for r in rows if r["class"] == c]), 3) for c in ("1-2", "3-5", ">5")}
            fp_in_clean = float(np.mean([r["fp"] for r in rows if r["n_ref"] == 0])) if any(r["n_ref"] == 0 for r in rows) else float("nan")
            table[name][cohort] = dict(cases=len(rows), lesions=int(sum(r["n_ref"] for r in rows)), f1=round(s["f1"], 3),
                                       precision=round(s["precision"], 3), sensitivity=round(s["sensitivity"], 3),
                                       fp_per_case=round(s["fp_per_case"], 2), median_case_f1=round(float(np.median(with_lesions)), 3),
                                       f1_by_count_class=by_class, fp_per_lesion_free_case=round(fp_in_clean, 2))
            e = table[name][cohort]
            print(f"{name:28s} {cohort:16s} n={e['cases']:3d} lesions={e['lesions']:4d}  F1 {e['f1']:.3f}  P {e['precision']:.2f}  "
                  f"S {e['sensitivity']:.2f}  FP/case {e['fp_per_case']:.2f}  median case F1 {e['median_case_f1']:.2f}  "
                  f"by count {by_class}  FP in lesion-free cases {e['fp_per_lesion_free_case']:.2f}", flush=True)
    # Paired comparison of every run against the FIRST run given (bootstrap over cases, 10 000 draws, same cases in both).
    names = [spec.split("=", 1)[0] for spec in args.run]
    rng = np.random.RandomState(0)
    for other in names[1:]:
        for cohort in sorted(set(per_case.get(names[0], {})) & set(per_case.get(other, {}))):
            a, b = per_case[other][cohort], per_case[names[0]][cohort]
            ids = sorted(set(a) & set(b)); A = [a[i] for i in ids]; B = [b[i] for i in ids]
            draws = []
            for _ in range(10000):
                ix = rng.randint(len(ids), size=len(ids)); draws.append(f1_of([A[j] for j in ix]) - f1_of([B[j] for j in ix]))
            lo, hi = np.percentile(draws, [2.5, 97.5]); delta = f1_of(A) - f1_of(B)
            table[other][cohort]["paired_vs_" + names[0]] = dict(delta_f1=round(delta, 3), ci95=[round(float(lo), 3), round(float(hi), 3)], n=len(ids))
            print(f"paired  {other} - {names[0]}  [{cohort}, n={len(ids)}]:  {delta:+.3f}  [{lo:+.3f}; {hi:+.3f}]", flush=True)
    if args.json:
        json.dump(dict(post_processing=dict(threshold=THRESHOLD, min_mm3=MIN_MM3), runs=table), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
