"""Score runs with the SUBJECT-LEVEL protocol used by recent VALDO Task 2 papers, to make numbers comparable.

Our project metric pools lesions over all cases (26-connectivity, touch rule). The VALDO challenge and CenSynCMB
(He et al., arXiv 2607.05325, 2026) score differently:
  * a predicted component matches a reference component when the centres of mass are less than 5 mm apart,
    one-to-one, closest first (code/metric_valdo.py, 6-connectivity as in the challenge paper);
  * F1 is computed PER SUBJECT; the challenge reports the median, CenSynCMB the mean over subjects.
How subjects WITHOUT reference lesions enter the mean is not stated in CenSynCMB, so three readings are printed:
lesion subjects only; all subjects with "nothing to find, nothing predicted" = 1 and any false alarm = 0.

The prediction is post-processed as everywhere in the project (threshold 0.3, >= 2 mm3, giant components removed;
with --synthseg also the CSF rule). NOT comparable one-to-one with published tables even so: they use all 72 cases and
three modalities, we use the 57 cross-validation cases (15 are held out) and T2* only.

    python -m cmb.analysis.subject_level_f1 --run NAME=FOLDER [...] --manifest M.json [--synthseg PATTERN] [--json out.json]
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.analysis.fp_location import ROOT, load_synthseg, synthseg_path
from cmb.analysis.missed_lesions import majority_label
from cmb.analysis.operating_point import CSF_LABELS
from cmb.transfer.evaluate import CONNECTIVITY_26, MAX_VOXELS, MIN_MM3, THRESHOLD, VOXEL_MM3

sys.path.insert(0, f"{ROOT}/code")
from metric_valdo import case_valdo            # noqa: E402


def post_process(probability: np.ndarray, seg) -> np.ndarray:
    labels, n = ndimage.label(probability > THRESHOLD, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= MIN_MM3) & (size <= MAX_VOXELS); keep[0] = False
    if seg is not None:
        for c, box in enumerate(ndimage.find_objects(labels), start=1):
            if keep[c] and majority_label(labels[box] == c, seg[box]) in CSF_LABELS:
                keep[c] = False
    return keep[labels]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="NAME=FOLDER")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--synthseg", help="path pattern with {id} or {sid}; if given, the CSF rule is applied")
    parser.add_argument("--json")
    args = parser.parse_args()
    cases = {e["id"]: e for p in args.manifest for e in json.load(open(p))}
    out = {}
    for spec in args.run:
        name, folder = spec.split("=", 1)
        folder = folder if os.path.isabs(folder) else f"{ROOT}/{folder}"
        rows = []
        for path in sorted(glob.glob(f"{folder}/fold*/*_pred_proba.nii.gz")):
            case_id = os.path.basename(path)[:-len("_pred_proba.nii.gz")]
            if case_id not in cases:
                continue
            image = nib.load(path)
            probability = np.asarray(image.dataobj).astype(np.float32) / 255.0
            reference = np.asarray(nib.load(f"{cases[case_id]['dir']}/{case_id}_label.nii.gz").dataobj) > 0
            seg = load_synthseg(synthseg_path(args.synthseg, case_id), image) if args.synthseg else None
            rows.append(case_valdo(reference, post_process(probability, seg), image.affine))
        with_lesions = [r for r in rows if r["n_ref"] > 0]
        without = [r for r in rows if r["n_ref"] == 0]
        f1_lesion_subjects = [r["f1"] for r in with_lesions]
        f1_all = f1_lesion_subjects + [1.0 if r["n_pred"] == 0 else 0.0 for r in without]
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        entry = dict(cases=len(rows), lesion_subjects=len(with_lesions),
                     mean_f1_lesion_subjects=round(float(np.mean(f1_lesion_subjects)), 3),
                     median_f1_lesion_subjects=round(float(np.median(f1_lesion_subjects)), 3),
                     mean_f1_all_subjects=round(float(np.mean(f1_all)), 3),
                     clean_subjects_without_false_alarm=f"{sum(r['n_pred'] == 0 for r in without)}/{len(without)}",
                     pooled_f1_centroid_rule=round(2 * tp / max(2 * tp + fp + fn, 1), 3),
                     recall=round(tp / max(tp + fn, 1), 3), precision=round(tp / max(tp + fp, 1), 3),
                     fp_per_subject=round(fp / len(rows), 2))
        out[name] = entry
        print(f"{name:12s} n={entry['cases']} ({entry['lesion_subjects']} with lesions)  per-subject F1: mean {entry['mean_f1_lesion_subjects']:.3f} / "
              f"median {entry['median_f1_lesion_subjects']:.3f} (lesion subjects), mean over all subjects {entry['mean_f1_all_subjects']:.3f}  |  "
              f"pooled F1 {entry['pooled_f1_centroid_rule']:.3f}  R {entry['recall']:.2f}  P {entry['precision']:.2f}  FP/subject {entry['fp_per_subject']:.2f}  "
              f"clean subjects without false alarm {entry['clean_subjects_without_false_alarm']}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
