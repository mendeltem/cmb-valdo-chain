"""Results per VALDO cohort, at the operating point each fold chose for itself.

The three VALDO cohorts differ mainly in slice thickness (cohort 1: 4 mm, cohort 2: 0.8 mm, cohort 3: 3-4 mm), so a pooled
score hides where a network wins or loses. This script reads the BINARY out-of-fold predictions a training run stored
(``fold<k>/<id>_pred.nii.gz``: threshold and minimum size selected on the inner fold) and pools lesion-level counts per
cohort -- raw and after the parameter-free CSF rule (component dropped if its majority SynthSeg label is CSF).
It reproduces the table of 2026-09-19 07:00 in protokoll.md (40-epoch tournament models), e.g. a03-aniso:
cohort 1 F1 0.500 / 0.510 with CSF rule, cohort 2 0.359 / 0.478, cohort 3 0.471 / 0.490.
For the FIXED operating point of the project use ``cmb.transfer.evaluate --split-valdo`` instead.

    python -m cmb.analysis.cohort_breakdown --run ergebnisse/turnier/a03-aniso ergebnisse/turnier/a01-attunet \
        --manifest dev/manifest_valdo_gitter_d.json --synthseg "dev/synthseg/{id}_synthseg.nii.gz"
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
from cmb.transfer.evaluate import CONNECTIVITY_26

sys.path.insert(0, f"{ROOT}/code")
from metric import hits            # noqa: E402


def without_csf(prediction: np.ndarray, seg: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(prediction, structure=CONNECTIVITY_26)
    keep = np.ones(n + 1, bool); keep[0] = False
    for c, box in enumerate(ndimage.find_objects(labels), start=1):
        if majority_label(labels[box] == c, seg[box]) in CSF_LABELS:
            keep[c] = False
    return keep[labels]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--synthseg", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    cases = {e["id"]: e for e in json.load(open(absolute(args.manifest)))}
    out = {}
    for run in args.run:
        rows = {}
        for path in sorted(glob.glob(f"{absolute(run)}/fold*/*_pred.nii.gz")):
            case_id = os.path.basename(path)[:-len("_pred.nii.gz")]
            if case_id not in cases:
                continue
            image = nib.load(path)
            prediction = np.asarray(image.dataobj) > 0
            reference = np.asarray(nib.load(f"{cases[case_id]['dir']}/{case_id}_label.nii.gz").dataobj) > 0
            seg = load_synthseg(synthseg_path(args.synthseg, case_id), image)
            rows.setdefault(case_id[4], []).append((hits(reference, prediction), hits(reference, without_csf(prediction, seg))))
        name = os.path.basename(absolute(run)); out[name] = {}
        for cohort in sorted(rows):
            entry = {}
            for label, k in (("raw", 0), ("csf_rule", 1)):
                tp, fp, fn = (sum(r[k][x] for r in rows[cohort]) for x in ("tp", "fp", "fn"))
                entry[label] = dict(f1=round(2 * tp / max(2 * tp + fp + fn, 1), 3), precision=round(tp / max(tp + fp, 1), 2),
                                    sensitivity=round(tp / max(tp + fn, 1), 2), fp_per_case=round(fp / len(rows[cohort]), 1))
            entry.update(cases=len(rows[cohort]), lesions=int(sum(r[0]["n_ref"] for r in rows[cohort])))
            out[name][cohort] = entry
            print(f"{name:22s} cohort {cohort}: {entry['cases']:2d} cases {entry['lesions']:3d} lesions   raw F1 {entry['raw']['f1']:.3f} P {entry['raw']['precision']:.2f} "
                  f"S {entry['raw']['sensitivity']:.2f} FP/case {entry['raw']['fp_per_case']:.1f}   with CSF rule F1 {entry['csf_rule']['f1']:.3f} P {entry['csf_rule']['precision']:.2f}")
    if args.json:
        json.dump(out, open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
