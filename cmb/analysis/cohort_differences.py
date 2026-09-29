"""Do the cohorts differ in their IMAGES, in their ANNOTATIONS, or in both?  (user question, 2026-09-21)

Joint training of several cohorts lowered every cohort's score (WORKLOG.md 5e, 5h). Two explanations are on the table:
the images differ (sequence, slice thickness, masks) or the experts marked differently (what counts as a microbleed, how
generously it is outlined). This script puts numbers on both, on the common training grid (0.5 x 0.5 x 1 mm, inverted
scale: lesions bright), per cohort:

  annotation   lesions per case (median, maximum); lesion volume in mm3 and in NATIVE voxels (volume / native voxel
               volume from info.json); share of lesions smaller than 4 native voxels;
               contrast = relative darkening against a ring 0.5-2 mm around the lesion (original intensity);
               outline index = how much of the lesion's brightness is still present in the first voxel ring (0.5 mm) OUTSIDE
               the outline, relative to a far ring 2-3 mm away: (shell - far ring) / (lesion - far ring). Near 1 = tight outline (the dark spot continues beyond
               the mask), near 0 = generous outline (the mask already contains the whole spot).
  image        median and interquartile range of the brain intensities on the inverted scale; "mimic load" = share of
               brain voxels at least as bright as the cohort's median lesion -- how many non-lesion voxels look like one.
Cohort = the manifest field, VALDO split into its three sub-cohorts. Sums only -- safe for the private cohorts.

    python -m cmb.analysis.cohort_differences --manifest dev/manifest_valdo_gitter_d.json /private/manifest_t2star.json /private/manifest_swi.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.analysis.missed_lesions import contrast_of

ROOT = "/home/uchralt/data/work/valdo-t2s"
IN_PLANE_3 = np.ones((3, 3, 1), bool)
IN_PLANE_7 = np.ones((7, 7, 1), bool)


def one_case(entry):
    case_id, folder = entry["id"], entry["dir"]
    image = np.asarray(nib.load(f"{folder}/{case_id}_image.nii.gz").dataobj, dtype=np.float32)
    label = np.asarray(nib.load(f"{folder}/{case_id}_label.nii.gz").dataobj) > 0
    info = json.load(open(f"{folder}/info.json")) if os.path.exists(f"{folder}/info.json") else {}
    native = info.get("quelle_spacing")                     # native voxel size written by the grid builder
    brain = image != 0
    labels, n = ndimage.label(label, structure=np.ones((3, 3, 3)))
    lesions = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        crop = tuple(slice(max(s.start - 8, 0), min(s.stop + 8, dim)) for s, dim in zip(box, labels.shape))
        mask = labels[crop] == index; others = label[crop] & ~mask; img = image[crop]
        shell = ndimage.binary_dilation(mask, structure=IN_PLANE_3) & ~mask & (img != 0) & ~others
        far = ndimage.binary_dilation(mask, structure=IN_PLANE_7, iterations=2) & ~ndimage.binary_dilation(mask, structure=IN_PLANE_3, iterations=3) & (img != 0) & ~others   # 2-3 mm away
        outline = None
        if shell.sum() >= 4 and far.sum() >= 8 and img[mask].mean() - img[far].mean() > 0.02:
            outline = float((img[shell].mean() - img[far].mean()) / (img[mask].mean() - img[far].mean()))
        lesions.append(dict(volume_mm3=float(mask.sum() * 0.25), contrast=contrast_of(img, mask, others), outline=outline,
                            brightness=float(img[mask].mean())))
    values = image[brain]
    quartiles = np.percentile(values, [25, 50, 75]) if values.size else [np.nan] * 3
    sample = values[:: max(values.size // 200000, 1)]                       # thinned brain intensities for the mimic load
    cohort = entry["cohort"]
    return dict(cohort=cohort, n=n, lesions=lesions, quartiles=[float(q) for q in quartiles], sample=sample.astype(np.float16), native=native)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--native-voxel-mm3", nargs="*", default=[], help="COHORT=mm3 if info.json does not carry the native spacing")
    parser.add_argument("--json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    entries = [e for m in args.manifest for e in json.load(open(absolute(m)))]
    native_given = {s.split("=")[0]: float(s.split("=")[1]) for s in args.native_voxel_mm3}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one_case, entries))
    out = {}
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    for cohort in sorted({r["cohort"] for r in results}):
        here = [r for r in results if r["cohort"] == cohort]
        lesions = [l for r in here for l in r["lesions"]]
        natives = [float(np.prod(r["native"])) for r in here if r["native"]]
        native_mm3 = native_given.get(cohort, med(natives) if natives else float("nan"))
        volumes = np.array([l["volume_mm3"] for l in lesions])
        brightness = med([l["brightness"] for l in lesions])
        sample = np.concatenate([r["sample"].astype(np.float32) for r in here])
        out[cohort] = dict(cases=len(here), lesions=len(lesions), lesions_per_case_median=med([r["n"] for r in here]), lesions_per_case_max=int(max(r["n"] for r in here)),
                           volume_mm3_median=round(med(volumes), 1), native_voxel_mm3=round(native_mm3, 2), native_voxels_median=round(med(volumes) / native_mm3, 1),
                           share_below_4_native_voxels=round(float((volumes / native_mm3 < 4).mean()), 2),
                           contrast_median=round(med([l["contrast"] for l in lesions if l["contrast"] is not None]), 3),
                           outline_index_median=round(med([l["outline"] for l in lesions if l["outline"] is not None]), 2),
                           brain_median=round(med([r["quartiles"][1] for r in here]), 3), brain_iqr=round(med([r["quartiles"][2] - r["quartiles"][0] for r in here]), 3),
                           mimic_load=round(float((sample >= brightness).mean()), 4))
        e = out[cohort]
        print(f"{cohort:16s} {e['cases']:3d} cases {e['lesions']:4d} lesions | per case median {e['lesions_per_case_median']:.0f} max {e['lesions_per_case_max']:3d} | volume {e['volume_mm3_median']:5.1f} mm3 = "
              f"{e['native_voxels_median']:4.1f} native voxels ({e['native_voxel_mm3']:.2f} mm3 each), < 4 voxels: {100 * e['share_below_4_native_voxels']:.0f} % | contrast {e['contrast_median']:.3f} | "
              f"outline index {e['outline_index_median']:.2f} | brain median {e['brain_median']:.2f} IQR {e['brain_iqr']:.2f} | mimic load {100 * e['mimic_load']:.1f} %")
    if args.json:
        json.dump(out, open(absolute(args.json), "w"), indent=1)


if __name__ == "__main__":
    main()
