"""How much lesion contrast is lost by resampling to the training grid?

Finding of 2026-09-19 (ARBEIT.md 4.8): label volume survives the resampling, lesion contrast does not. Measured per
reference lesion (26-connected component), before (native grid, e.g. dev/preproc_roh) and after (training grid):
  contrast       = mean inside the lesion - mean in a ring around it      (inverted image: lesions are bright)
  ring           = voxels of the SAME slices whose in-plane distance to the lesion is > 0.5 mm and <= 2.0 mm, inside the
                   brain (image != 0), other lesions excluded. The ring is defined in millimetres, not voxels -- a shell of
                   n voxels is not comparable between a 0.45 x 0.45 x 4 mm and a 0.5 x 0.5 x 1 mm grid (first attempt,
                   discarded).
  peak contrast  = mean of the brightest tenth of the lesion voxels - ring mean
Reported per cohort over the cross-validation cases: mean contrast before -> after and the relative change of that mean,
the same for the peak, and the change of the summed lesion volume.
Documented (dev/preproc_roh -> dev/gitter_d, 57 cases): contrast 0.148 -> 0.114 (-23 %), 0.123 -> 0.109 (-12 %),
0.128 -> 0.087 (-32 %); peak -20 / -11 / -22 %; volume -1 / 0 / 0 %.

    python -m cmb.analysis.resampling_contrast --source dev/preproc_roh --grid dev/gitter_d --manifest dev/manifest_valdo_gitter_d.json
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
RING_MM = (0.5, 2.0)


def lesion_contrasts(folder: str, case_id: str):
    """-> list of (contrast, peak contrast) per lesion, summed lesion volume in mm3."""
    image_nii = nib.load(f"{folder}/{case_id}_image.nii.gz")
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    label = np.asarray(nib.load(f"{folder}/{case_id}_label.nii.gz").dataobj) > 0
    zooms = [float(z) for z in image_nii.header.get_zooms()[:3]]
    labels, n = ndimage.label(label, structure=np.ones((3, 3, 3)))
    margin = [int(math.ceil(RING_MM[1] / zooms[0])) + 1, int(math.ceil(RING_MM[1] / zooms[1])) + 1, 0]
    rows = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        crop = tuple(slice(max(s.start - m, 0), min(s.stop + m, dim)) for s, m, dim in zip(box, margin, labels.shape))
        mask = labels[crop] == index
        ring = np.zeros(mask.shape, bool)
        for z in range(mask.shape[2]):                          # in-plane distance, slice by slice
            if mask[:, :, z].any():
                distance = ndimage.distance_transform_edt(~mask[:, :, z], sampling=zooms[:2])
                ring[:, :, z] = (distance > RING_MM[0]) & (distance <= RING_MM[1])
        ring &= (image[crop] != 0) & ~label[crop]
        if ring.sum() < 4:
            continue
        inside = np.sort(image[crop][mask])
        top = inside[-max(int(round(len(inside) / 10)), 1):]
        ring_mean = float(image[crop][ring].mean())
        rows.append((float(inside.mean()) - ring_mean, float(top.mean()) - ring_mean))
    return rows, float(label.sum() * np.prod(zooms))


def one_case(job):
    case_id, source, grid = job
    before, volume_before = lesion_contrasts(f"{source}/{case_id}", case_id)
    after, volume_after = lesion_contrasts(f"{grid}/{case_id}", case_id)
    return case_id[4], before, after, volume_before, volume_after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="native-grid folder, e.g. dev/preproc_roh")
    parser.add_argument("--grid", required=True, help="training grid, e.g. dev/gitter_d")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    cases = [e["id"] for e in json.load(open(absolute(args.manifest))) if e["n_cmb"] > 0]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one_case, [(c, absolute(args.source), absolute(args.grid)) for c in cases]))
    out = {}
    for cohort in sorted({r[0] for r in results}):
        here = [r for r in results if r[0] == cohort]
        before = [x for r in here for x in r[1]]; after = [x for r in here for x in r[2]]
        mean = lambda rows, k: float(np.mean([x[k] for x in rows]))
        change = lambda a, b: 100.0 * (b - a) / a
        out[cohort] = dict(cases=len(here), lesions_before=len(before), lesions_after=len(after),
                           contrast_before=round(mean(before, 0), 3), contrast_after=round(mean(after, 0), 3),
                           contrast_change_pct=round(change(mean(before, 0), mean(after, 0)), 1),
                           peak_change_pct=round(change(mean(before, 1), mean(after, 1)), 1),
                           volume_change_pct=round(change(sum(r[3] for r in here), sum(r[4] for r in here)), 1))
        e = out[cohort]
        print(f"cohort {cohort}: {e['cases']:2d} cases, {e['lesions_before']} / {e['lesions_after']} lesions   contrast {e['contrast_before']:.3f} -> "
              f"{e['contrast_after']:.3f} ({e['contrast_change_pct']:+.0f} %)   peak {e['peak_change_pct']:+.0f} %   lesion volume {e['volume_change_pct']:+.0f} %")
    if args.json:
        json.dump(out, open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
