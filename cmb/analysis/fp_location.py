"""Where in the brain do false alarms and misses sit?  (idea from the literature scan, 2026-09-20)

For every predicted component and every reference lesion of a run the anatomical location is read from the
SynthSeg map (majority label under the component, see code/gewebekarte.py): lobar, deep, infratentorial,
mesiotemporal, CSF, outside. Predictions are post-processed exactly as in cmb.transfer.evaluate
(threshold 0.3, >= 2 mm3, giant components removed), so the counts match the tables there.

Output per cohort:
  * per location: predictions, true/false positives, precision; reference lesions, detected, sensitivity
  * "what if all predictions at location X were dropped": pooled F1 before/after. This is EXPLORATORY --
    a rule picked this way is selected on the evaluated cases and must be confirmed with cross-fold selection
    before it counts as a result.
  * the same what-if for a stricter probability threshold at one location only.

    python -m cmb.analysis.fp_location --run ergebnisse/turnier/a03-aniso-e60-gd \
        --manifest dev/manifest_valdo_gitter_d.json --synthseg "dev/synthseg/{id}_synthseg.nii.gz"

`{id}` is the case id; `{sid}` is the id without the cohort prefix (c2s-/csw-) used for the private grids.
Sums only -- safe to show for the private cohort.
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import nibabel as nib
import nibabel.processing
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
from gewebekarte import ORTE, ort_der_komponente      # noqa: E402

from cmb.transfer.evaluate import CONNECTIVITY_26, MAX_VOXELS, MIN_MM3, THRESHOLD, VOXEL_MM3   # noqa: E402

ENGLISH = {"lobaer": "lobar", "tief": "deep", "infratentoriell": "infratentorial", "mesiotemporal": "mesiotemporal",
           "liquor": "csf", "ausserhalb": "outside"}
STRICT_THRESHOLDS = (0.5, 0.7, 0.9)


def components(mask: np.ndarray):
    labels, n = ndimage.label(mask, structure=CONNECTIVITY_26)
    return labels, n, ndimage.find_objects(labels)


def padded(box, shape, margin: int = 4):
    return tuple(slice(max(s.start - margin, 0), min(s.stop + margin, dim)) for s, dim in zip(box, shape))


def load_synthseg(path: str, like: nib.Nifti1Image) -> np.ndarray:
    seg = nib.load(path)
    if seg.shape != like.shape:                       # private cohorts: SynthSeg ran on the native 1 mm grid
        seg = nibabel.processing.resample_from_to(seg, like, order=0)
    return np.asarray(seg.dataobj).astype(np.int32)


def synthseg_path(pattern: str, case_id: str) -> str:
    """Fill {id} / {sid} in the pattern; {sid} is the id without the private cohort prefix (c2s-, csw-)."""
    short = case_id.split("-", 1)[1] if case_id[:4] in ("c2s-", "csw-") else case_id
    path = pattern.format(id=case_id, sid=short)
    return path if os.path.isabs(path) else f"{ROOT}/{path}"


def f1(tp_ref: int, fp: int, fn: int) -> float:
    return 2 * tp_ref / (2 * tp_ref + fp + fn) if tp_ref + fp + fn else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="result folder with fold*/<id>_pred_proba.nii.gz")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--synthseg", required=True, help="path pattern with {id} or {sid}")
    parser.add_argument("--json", help="write the tables here as well")
    args = parser.parse_args()
    folder = args.run if os.path.isabs(args.run) else f"{ROOT}/{args.run}"
    cases = {e["id"]: e for p in args.manifest for e in json.load(open(p))}

    predictions: List[Dict] = []       # one row per predicted component: location, hit, max probability
    references: List[Dict] = []        # one row per reference lesion: location, detected, best probability on it
    missing = 0
    for path in sorted(glob.glob(f"{folder}/fold*/*_pred_proba.nii.gz")):
        case_id = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
        if case_id not in cases:
            continue
        seg_path = synthseg_path(args.synthseg, case_id)
        if not os.path.exists(seg_path):
            missing += 1
            continue
        image = nib.load(path)
        probability = np.asarray(image.dataobj).astype(np.float32) / 255.0
        reference = np.asarray(nib.load(f"{cases[case_id]['dir']}/{case_id}_label.nii.gz").dataobj) > 0
        seg = load_synthseg(seg_path, image)

        labels, n, boxes = components(probability > THRESHOLD)
        size = np.bincount(labels.ravel(), minlength=n + 1)
        kept = np.zeros(probability.shape, bool)
        for c in range(1, n + 1):
            if size[c] * VOXEL_MM3 < MIN_MM3 or size[c] > MAX_VOXELS:
                continue
            box = padded(boxes[c - 1], labels.shape)
            mask = labels[box] == c
            kept[box] |= mask
            predictions.append(dict(case=case_id, location=ort_der_komponente(mask, seg[box]),
                                    hit=bool((reference[box] & mask).any()), p_max=float(probability[box][mask].max())))
        ref_labels, n_ref, ref_boxes = components(reference)
        for c in range(1, n_ref + 1):
            box = padded(ref_boxes[c - 1], ref_labels.shape)
            mask = ref_labels[box] == c
            references.append(dict(case=case_id, location=ort_der_komponente(mask, seg[box]),
                                   detected=bool((kept[box] & mask).any()), p_max=float(probability[box][mask].max())))

    n_cases = len({r["case"] for r in predictions} | {r["case"] for r in references})
    print(f"{len(predictions)} predictions, {len(references)} reference lesions, {n_cases} cases with either"
          + (f", {missing} cases skipped (no SynthSeg map)" if missing else ""))
    fp_all = sum(not p["hit"] for p in predictions)
    det_all = sum(r["detected"] for r in references); fn_all = len(references) - det_all
    base = f1(det_all, fp_all, fn_all)
    print(f"baseline pooled F1 {base:.3f}  (detected {det_all}, FP {fp_all}, missed {fn_all})\n")

    table = {}
    print(f"{'location':16s} {'pred':>5s} {'TP':>4s} {'FP':>4s} {'prec':>5s} | {'ref':>4s} {'det':>4s} {'sens':>5s} | "
          f"{'F1 if dropped':>13s} | " + "  ".join(f"F1 thr {t}" for t in STRICT_THRESHOLDS))
    for german in ORTE:
        here = [p for p in predictions if p["location"] == german]
        refs = [r for r in references if r["location"] == german]
        tp = sum(p["hit"] for p in here); fp = len(here) - tp; det = sum(r["detected"] for r in refs)
        # what-if 1: drop every prediction here -> its FP vanish, its detected lesions become misses
        dropped = f1(det_all - det, fp_all - fp, fn_all + det)
        # what-if 2: stricter threshold here only. A lesion stays detected if the probability ON it reaches the threshold
        # (approximation: the component could still fail the size rule); an FP stays if its maximum reaches it.
        strict = {}
        for t in STRICT_THRESHOLDS:
            fp_left = sum(1 for p in here if not p["hit"] and p["p_max"] >= t)
            det_left = sum(1 for r in refs if r["detected"] and r["p_max"] >= t)
            strict[t] = f1(det_all - det + det_left, fp_all - fp + fp_left, fn_all + det - det_left)
        row = dict(predictions=len(here), tp=tp, fp=fp, precision=round(tp / len(here), 3) if here else None,
                   reference=len(refs), detected=det, sensitivity=round(det / len(refs), 3) if refs else None,
                   f1_if_dropped=round(dropped, 3), f1_strict_threshold={str(t): round(v, 3) for t, v in strict.items()})
        table[ENGLISH[german]] = row
        print(f"{ENGLISH[german]:16s} {len(here):5d} {tp:4d} {fp:4d} {(tp / len(here) if here else float('nan')):5.2f} | "
              f"{len(refs):4d} {det:4d} {(det / len(refs) if refs else float('nan')):5.2f} | {dropped:13.3f} | "
              + "  ".join(f"{strict[t]:10.3f}" for t in STRICT_THRESHOLDS))
    if args.json:
        json.dump(dict(run=folder, baseline_f1=round(base, 3), by_location=table,
                       note="what-if columns are exploratory; confirm any rule with cross-fold selection"),
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
