"""Write the manifest of the VALDO cross-validation cases for one training grid.

A manifest is the list every tool of this package reads: one entry per case with its id, its fold, the folder holding
``<id>_image.nii.gz``, ``<id>_frst.nii.gz`` and ``<id>_label.nii.gz``, the cohort and the number of reference lesions
(26-connected components on that grid). Folds come from ``dev/cases.json`` (written by ``code/split.py``, seed 0);
the 15 held-out cases (fold -1) are left out on purpose -- they are touched only once, at the very end.
``--held-out`` writes exactly those 15 instead; they get the AUXILIARY fold 0 because every loader only accepts folds 0..4
(it is a folder name for the prediction tools, nothing is trained on it); used once, 2026-09-28.

    python -m cmb.transfer.build_valdo_manifest --grid dev/gitter_d --out dev/manifest_valdo_gitter_d.json
    python -m cmb.transfer.build_valdo_manifest --grid dev/gitter_d --held-out --out dev/manifest_valdo_gitter_d_heldout.json
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--held-out", action="store_true", help="the 15 held-out cases (fold -1) instead of the CV cases")
    args = parser.parse_args()
    grid = args.grid if os.path.isabs(args.grid) else f"{ROOT}/{args.grid}"
    out = args.out if os.path.isabs(args.out) else f"{ROOT}/{args.out}"
    entries = []
    for case in json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]:
        if args.held_out != (case["fold"] == -1):
            continue
        case_id = case["id"]
        label = np.asarray(nib.load(f"{grid}/{case_id}/{case_id}_label.nii.gz").dataobj) > 0
        entries.append(dict(id=case_id, fold=0 if args.held_out else case["fold"], dir=f"{grid}/{case_id}", cohort=f"valdo-{case_id[4]}",
                            n_cmb=int(ndimage.label(label, structure=np.ones((3, 3, 3)))[1])))
    json.dump(entries, open(out, "w"), indent=1)
    print(f"{len(entries)} cases, {sum(e['n_cmb'] for e in entries)} lesions -> {out}")


if __name__ == "__main__":
    main()
