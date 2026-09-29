"""Compare models at THEIR OWN honest operating point  (threshold x minimum size, cross-fold selection).

Why: the project's fixed post-processing (threshold 0.3, at least 2 mm3) was selected on the baseline model. A model
trained with another loss is calibrated differently -- it may need a stricter threshold -- so comparing both at the
baseline's cell can be unfair to the newcomer. Picking the best cell per model on all cases would be unfair the other
way round (optimistic). Hence, per model:

  * every out-of-fold probability map is scored for every cell of the grid (project metric: 26-connectivity, touch rule;
    giant components removed; with --synthseg also the parameter-free CSF rule);
  * for fold k the cell with the best pooled F1 over the OTHER four folds is applied to fold k (cross-fold selection);
    the pooled score over all cases with these five choices is the honest number;
  * next to it: the fixed project cell and the best single cell (optimistic, for orientation only).
Runs are then compared pairwise on their cross-selected results (bootstrap over cases, 10 000 draws) against the FIRST run.

    python -m cmb.analysis.operating_point --run NAME=FOLDER [NAME=FOLDER ...] --manifest M.json \
        [--synthseg "dev/synthseg/{id}_synthseg.nii.gz"] [--json out.json] [--workers 6]
Sums only -- safe for the private cohort.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.analysis.fp_location import ROOT, load_synthseg, synthseg_path
from cmb.analysis.missed_lesions import majority_label
from cmb.transfer.evaluate import CONNECTIVITY_26, MAX_VOXELS, MIN_MM3, THRESHOLD, VOXEL_MM3

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SIZES_MM3 = (1.0, 2.0, 4.0, 6.0, 8.0)
CSF_LABELS = {4, 5, 14, 15, 24, 43, 44}
Cell = Tuple[float, float]
CELLS: List[Cell] = [(t, m) for t in THRESHOLDS for m in MIN_SIZES_MM3]
FIXED: Cell = (THRESHOLD, MIN_MM3)


def score_case(job) -> Dict:
    """-> {cell: (tp, fp, fn)} for one case, with the semantics of metric.hits."""
    case_id, case_dir, fold, proba_path, seg_path = job
    image = nib.load(proba_path)
    probability = np.asarray(image.dataobj).astype(np.float32) / 255.0
    ref_labels, n_ref = ndimage.label(np.asarray(nib.load(f"{case_dir}/{case_id}_label.nii.gz").dataobj) > 0, structure=CONNECTIVITY_26)
    seg = load_synthseg(seg_path, image) if seg_path else None
    cells = {}
    for t in THRESHOLDS:
        labels, n = ndimage.label(probability > t, structure=CONNECTIVITY_26)
        size = np.bincount(labels.ravel(), minlength=n + 1)
        usable = size <= MAX_VOXELS; usable[0] = False
        if seg is not None:                                   # CSF rule: drop components whose majority SynthSeg label is CSF
            for c, box in enumerate(ndimage.find_objects(labels), start=1):
                if usable[c] and majority_label(labels[box] == c, seg[box]) in CSF_LABELS:
                    usable[c] = False
        both = (ref_labels > 0) & (labels > 0)
        pairs = np.unique(np.stack([ref_labels[both], labels[both]], axis=1), axis=0) if both.any() else np.zeros((0, 2), int)
        for m in MIN_SIZES_MM3:
            keep = usable & (size * VOXEL_MM3 >= m)
            kept_pairs = pairs[keep[pairs[:, 1]]] if len(pairs) else pairs
            tp = len(set(kept_pairs[:, 0].tolist()))
            fp = int(keep.sum()) - len(set(kept_pairs[:, 1].tolist()))
            cells[(t, m)] = (tp, fp, n_ref - tp)
    return dict(case=case_id, fold=fold, cells=cells)


def pooled_f1(rows: List[Tuple[int, int, int]]) -> float:
    tp, fp, fn = (sum(r[i] for r in rows) for i in range(3))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="NAME=FOLDER")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--synthseg", help="path pattern with {id} or {sid}; if given, the CSF rule is applied")
    parser.add_argument("--json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cases = {e["id"]: e for p in args.manifest for e in json.load(open(p))}

    selected: Dict[str, Dict[str, Tuple[int, int, int]]] = {}       # run -> case -> (tp, fp, fn) at the cross-selected cell
    out: Dict = dict(grid=dict(thresholds=THRESHOLDS, min_sizes_mm3=MIN_SIZES_MM3), csf_rule=bool(args.synthseg), runs={})
    for spec in args.run:
        name, folder = spec.split("=", 1)
        folder = folder if os.path.isabs(folder) else f"{ROOT}/{folder}"
        jobs = []
        for path in sorted(glob.glob(f"{folder}/fold*/*_pred_proba.nii.gz")):
            case_id = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
            if case_id in cases:
                seg_path = synthseg_path(args.synthseg, case_id) if args.synthseg else None
                jobs.append((case_id, cases[case_id]["dir"], cases[case_id]["fold"], path, seg_path))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(score_case, jobs))

        table = {cell: pooled_f1([r["cells"][cell] for r in results]) for cell in CELLS}
        best = max(CELLS, key=lambda cell: table[cell])
        chosen, rows = {}, {}
        for k in sorted({r["fold"] for r in results}):
            others = [r for r in results if r["fold"] != k]
            chosen[k] = max(CELLS, key=lambda cell: pooled_f1([r["cells"][cell] for r in others]))
            for r in results:
                if r["fold"] == k:
                    rows[r["case"]] = r["cells"][chosen[k]]
        selected[name] = rows
        tp, fp, fn = (sum(r[i] for r in rows.values()) for i in range(3))
        entry = dict(cases=len(results), fixed_cell_f1=round(table[FIXED], 3), best_single_cell=dict(threshold=best[0], min_mm3=best[1], f1=round(table[best], 3)),
                     cross_selected=dict(f1=round(pooled_f1(list(rows.values())), 3), precision=round(tp / max(tp + fp, 1), 3),
                                         sensitivity=round(tp / max(tp + fn, 1), 3), fp_per_case=round(fp / len(results), 2),
                                         cells_per_fold={str(k): list(v) for k, v in chosen.items()}),
                     f1_grid={f"{t}/{m}": round(table[(t, m)], 3) for t, m in CELLS})
        out["runs"][name] = entry
        print(f"\n{name}  ({len(results)} cases{', CSF rule' if args.synthseg else ''})")
        print("threshold  " + "  ".join(f">={m:g} mm3" for m in MIN_SIZES_MM3))
        for t in THRESHOLDS:
            print(f"   {t:4.1f}     " + "  ".join(f"{table[(t, m)]:8.3f}" for m in MIN_SIZES_MM3))
        c = entry["cross_selected"]
        print(f"fixed cell {FIXED}: {table[FIXED]:.3f}   best single cell {best}: {table[best]:.3f} (optimistic)")
        print(f"CROSS-SELECTED: F1 {c['f1']:.3f}  P {c['precision']:.2f}  S {c['sensitivity']:.2f}  FP/case {c['fp_per_case']:.2f}   cells per fold {list(chosen.values())}")

    names = list(selected)
    rng = np.random.RandomState(0)
    for other in names[1:]:
        ids = sorted(set(selected[other]) & set(selected[names[0]]))
        a = [selected[other][i] for i in ids]; b = [selected[names[0]][i] for i in ids]
        draws = []
        for _ in range(10000):
            ix = rng.randint(len(ids), size=len(ids))
            draws.append(pooled_f1([a[j] for j in ix]) - pooled_f1([b[j] for j in ix]))
        lo, hi = np.percentile(draws, [2.5, 97.5]); delta = pooled_f1(a) - pooled_f1(b)
        out["runs"][other]["paired_vs_" + names[0]] = dict(delta_f1=round(delta, 3), ci95=[round(float(lo), 3), round(float(hi), 3)], n=len(ids))
        print(f"\npaired (cross-selected)  {other} - {names[0]}  [n={len(ids)}]:  {delta:+.3f}  [{lo:+.3f}; {hi:+.3f}]")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
