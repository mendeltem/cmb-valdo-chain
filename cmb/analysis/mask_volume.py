"""How tight is the delivered brain mask?  Mask volume per VALDO cohort and the share of it that is not brain.

Finding of 2026-09-19 (ARBEIT.md 4.2): VALDO ships "masked" images, but only cohort 1 is skull-stripped; cohort 3 keeps
eyes, skull base and scalp. Measured here exactly as then:
  mask        = image != 0 on the training grid
  volume      = voxels x voxel volume, in ml
  not brain   = mask voxels where the SynthSeg map has label 0, as a share of the mask
Reported per cohort (first digit of the case number): mean, smallest and largest volume, mean share outside the brain.
Documented values (grid C, 72 cases): 1419 / 1770 / 2533 ml (cohort 3: 2085-3009), outside 4 % / 16 % / 40 %.

    python -m cmb.analysis.mask_volume --grid dev/gitter_d --synthseg "dev/synthseg/{id}_synthseg.nii.gz"
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

from cmb.analysis.fp_location import ROOT, load_synthseg, synthseg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--synthseg", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()
    grid = args.grid if os.path.isabs(args.grid) else f"{ROOT}/{args.grid}"
    cohorts = {}
    for folder in sorted(glob.glob(f"{grid}/sub-*")):
        case_id = os.path.basename(folder)
        image = nib.load(f"{folder}/{case_id}_image.nii.gz")
        mask = np.asarray(image.dataobj) != 0
        seg = load_synthseg(synthseg_path(args.synthseg, case_id), image)
        volume_ml = float(mask.sum() * np.prod(image.header.get_zooms()[:3]) / 1000.0)
        cohorts.setdefault(case_id[4], []).append((volume_ml, float((mask & (seg == 0)).sum() / mask.sum())))
    out = {}
    for cohort, rows in sorted(cohorts.items()):
        volumes = [r[0] for r in rows]; outside = [r[1] for r in rows]
        out[cohort] = dict(cases=len(rows), mean_ml=round(float(np.mean(volumes))), min_ml=round(min(volumes)), max_ml=round(max(volumes)),
                           mean_share_outside_brain=round(float(np.mean(outside)), 3))
        e = out[cohort]
        print(f"cohort {cohort}: {e['cases']:2d} cases  mask {e['mean_ml']} ml ({e['min_ml']}-{e['max_ml']})  outside the SynthSeg brain {100 * e['mean_share_outside_brain']:.0f} %")
    if args.json:
        json.dump(out, open(args.json if os.path.isabs(args.json) else f"{ROOT}/{args.json}", "w"), indent=1)


if __name__ == "__main__":
    main()
