"""wk0a step 4: hole statistics of the model grids, old vs v2 (aggregates only, no case ids).

Same definition as qc/build_model_input_qc.py holes_of (3D fill plus axial 2D fill, brain =
image > 0 on the model grid). Prints per cohort: cases with holes, median hole volume,
reference CMB total and CMB touching a hole; plus, for the v2 cohorts, WHERE the remaining
in-hole CMB voxels are (distance to the brain edge in mm).

    python -m cmb.maskfix.qc_summary
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from cmb.paths import PRIVATE
N26 = np.ones((3, 3, 3), bool)
COHORTS = {
    "swi": f"{PRIVATE}/gitter/manifest_swi_mix.json",
    "swi_v2": f"{PRIVATE}/gitter/manifest_swi_mix_v2.json",
    "t2star": f"{PRIVATE}/gitter/manifest_t2star.json",
    "t2star_v2": f"{PRIVATE}/gitter/manifest_t2star_v2.json",
}


def holes_of(brain):
    filled = ndimage.binary_fill_holes(brain)
    filled |= np.stack([ndimage.binary_fill_holes(brain[..., k]) for k in range(brain.shape[2])], -1)
    return filled & ~brain


def one_cohort(name):
    rows = json.load(open(COHORTS[name]))
    n_hole_cases = 0
    hole_ml, n_cmb, cmb_in_holes = [], 0, 0
    edge_dist = []
    for e in rows:
        d = e["dir"]
        img = np.asarray(nib.load(f"{d}/{e['id']}_image.nii.gz").dataobj, np.float32)
        lab = np.asarray(nib.load(f"{d}/{e['id']}_label.nii.gz").dataobj) > 0
        zooms = nib.load(f"{d}/{e['id']}_image.nii.gz").header.get_zooms()[:3]
        vox_ml = float(np.prod(zooms)) / 1000
        brain = img > 0
        hole = holes_of(brain)
        cl, n = ndimage.label(lab, N26)
        in_hole = set(np.unique(cl[lab & ~brain])) - {0}
        if hole.sum() > 0:
            n_hole_cases += 1
        hole_ml.append(hole.sum() * vox_ml)
        n_cmb += int(n)
        cmb_in_holes += len(in_hole)
        if name.endswith("_v2") and in_hole:
            # where are the remaining in-hole CMBs: in a hole (enclosed) or outside the
            # brain (touching the border)? distance of the hole voxels to the border, in mm
            border_dist = ndimage.distance_transform_edt(hole) * zooms[0]
            for c in in_hole:
                m = lab & (cl == c)
                in_true_hole = hole[m]
                out_edge = ~in_true_hole
                d = float(border_dist[m & in_true_hole].max()) if in_true_hole.any() else None
                edge_dist.append(dict(
                    in_true_hole=bool(in_true_hole.any()),
                    outside_brain=bool(out_edge.any()),
                    max_hole_depth_mm=round(d, 1) if d is not None else None))
    med = float(np.median(hole_ml)) if hole_ml else 0.0
    out = dict(cohort=name, cases=len(rows), cases_with_holes=n_hole_cases,
               median_holes_ml=round(med, 3), n_cmb=n_cmb, cmb_in_holes=cmb_in_holes,
               pct_in_holes=round(100.0 * cmb_in_holes / max(n_cmb, 1), 1))
    if name.endswith("_v2"):
        out["v2_remaining_detail"] = edge_dist
    return out


def main():
    res = [one_cohort(c) for c in ("swi", "swi_v2", "t2star", "t2star_v2")]
    for r in res:
        print(json.dumps(r))
    json.dump(res, open(f"{PRIVATE}/maskfix/qc_summary.json", "w"), indent=1)
    print(f"written {PRIVATE}/maskfix/qc_summary.json")


if __name__ == "__main__":
    main()
