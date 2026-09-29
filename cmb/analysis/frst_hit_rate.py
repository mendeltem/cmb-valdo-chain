"""Does the radial-symmetry channel (FRST) light up at the microbleeds?  Hit rate per grid and cohort.

Definition as in code/gitter_c_bauen.py::frst_guete (used for the finding of 2026-09-19, WORKLOG.md 4.1 / 4.11):
  brain   = image > 0 with holes filled slice by slice (code/frst.py::hirnmaske)
  hit     = the maximum of the FRST map inside a reference lesion (26-connected component) reaches the 99th percentile
            of the FRST values of that case's brain
All 72 cases of a grid are counted (the FRST map uses no labels, so the held-out cases leak nothing).
Documented: grid C (original vessel inpainting) 201/236 = 85 % (82/106, 92/96, 27/34 per cohort);
            grid D (no inpainting) 227/236 = 96 % (100/106, 93/96, 34/34).

    python -m cmb.analysis.frst_hit_rate --grid dev/gitter_c dev/gitter_d
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
from frst import hirnmaske            # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", nargs="+", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()
    out = {}
    for grid_arg in args.grid:
        grid = grid_arg if os.path.isabs(grid_arg) else f"{ROOT}/{grid_arg}"
        hits, totals = {}, {}
        for folder in sorted(glob.glob(f"{grid}/sub-*")):
            case_id = os.path.basename(folder); cohort = case_id[4]
            load = lambda kind: np.asarray(nib.load(f"{folder}/{case_id}_{kind}.nii.gz").dataobj)
            labels, n = ndimage.label(load("label") > 0, structure=np.ones((3, 3, 3)))
            if n == 0:
                continue
            frst = load("frst").astype(np.float32)
            brain = hirnmaske(load("image").astype(np.float32))
            q99 = float(np.percentile(frst[brain], 99))
            peak = np.asarray(ndimage.maximum(frst, labels, index=np.arange(1, n + 1)))
            hits[cohort] = hits.get(cohort, 0) + int((peak >= q99).sum()); totals[cohort] = totals.get(cohort, 0) + n
        name = os.path.basename(grid)
        out[name] = dict(hits=sum(hits.values()), lesions=sum(totals.values()), per_cohort={c: [hits[c], totals[c]] for c in sorted(totals)})
        print(f"{name}: {out[name]['hits']}/{out[name]['lesions']} = {100 * out[name]['hits'] / out[name]['lesions']:.0f} %   per cohort "
              + ", ".join(f"{c}: {hits[c]}/{totals[c]}" for c in sorted(totals)))
    if args.json:
        json.dump(out, open(args.json if os.path.isabs(args.json) else f"{ROOT}/{args.json}", "w"), indent=1)


if __name__ == "__main__":
    main()
