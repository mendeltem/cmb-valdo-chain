#!/usr/bin/env python3
"""we2 grid: build the two input variants.

A "native":  original grid per case (dev/preproc/<id>/), patch 96x64x48.
B "shared":  all cases resampled to 0.5 x 0.5 x 1.0 mm (image linear,
                FRST linear, label nearest), patch (96,64,48) thus spans
                physically 48 x 32 x 48 mm -- in cohort 2 (z=0.8 mm) the
                same volume as A (48 voxels * 0.8 mm = 38.4 mm ~ 48 mm).

Only folds 0-4 (training+test), validation (fold -1) stays outside.
Resumable: finished cases (all 4 files present) are skipped.
"""
import os, sys, json, glob
import numpy as np
import SimpleITK as sitk
import nibabel as nib

ROOT = "/home/uchralt/data/work/valdo-t2s"
ZIEL = [0.5, 0.5, 1.0]  # mm
PATCH_B = (96, 64, 48)  # (x, y, z) voxels
PATCH_MM_B = [PATCH_B[0] * ZIEL[0], PATCH_B[1] * ZIEL[1], PATCH_B[2] * ZIEL[2]]
PATCH_A_MM = [96 * 0.48828125, 64 * 0.48828125, 48 * 0.8]  # in cohort 2, patch A

def resample(src, dst, spacing, interp):
    """Resample onto the target spacing; physical extent and direction stay
    unchanged. SimpleITK 2.5.6: old Resample API (no ResampleImage)."""
    img = sitk.ReadImage(src)
    osize = img.GetSize()
    osp = img.GetSpacing()
    nsize = [int(round(osize[i] * osp[i] / spacing[i])) for i in range(3)]
    ref = sitk.Image(nsize, img.GetDimension())
    ref.SetSpacing(spacing)
    ref.SetOrigin(img.GetOrigin())
    ref.SetDirection(img.GetDirection())
    out = sitk.Resample(img, ref, sitk.Transform(), interp,
                        0.0, img.GetPixelID())
    sitk.WriteImage(out, dst)
    return out

def main():
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    cases = json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]
    fold0 = sorted({c["id"] for c in cases if c["fold"] == 0})
    info = {"ziel_spacing_mm": ZIEL, "patch_b_voxel": list(PATCH_B),
            "patch_b_mm": PATCH_MM_B, "patch_a_mm_ko2": PATCH_A_MM,
            "faelle_gesamt": 0, "faelle_b": 0, "shapes_b": {}, "spacing_faktoren": {}}
    n = 0
    for c in sorted(cases, key=lambda z: z["id"]):
        if c["fold"] == -1:
            continue
        if limit is not None and n >= limit:
            break
        pid = c["id"]
        n += 1
        src = f"{ROOT}/dev/preproc/{pid}"
        dst = f"{ROOT}/dev/gitter_b/{pid}"
        os.makedirs(dst, exist_ok=True)
        files = {
            "image": f"{src}/{pid}_image.nii.gz",
            "frst": f"{src}/{pid}_frst.nii.gz",
            "label": f"{src}/{pid}_label.nii.gz",
        }
        done = all(os.path.exists(f"{dst}/{pid}_{k}.nii.gz") for k in files)
        if done:
            continue
        h = nib.load(files["image"]).header
        sp = [float(v) for v in h.get_zooms()]
        faktor = [round(sp[i] / ZIEL[i], 4) for i in range(3)]
        resample(files["image"], f"{dst}/{pid}_image.nii.gz", ZIEL, sitk.sitkLinear)
        resample(files["frst"], f"{dst}/{pid}_frst.nii.gz", ZIEL, sitk.sitkLinear)
        resample(files["label"], f"{dst}/{pid}_label.nii.gz", ZIEL, sitk.sitkNearestNeighbor)
        shp = sitk.ReadImage(f"{dst}/{pid}_image.nii.gz").GetSize()  # (x,y,z)
        info["faelle_b"] += 1
        key = (tuple(shp), tuple(round(f, 4) for f in faktor))
        info["shapes_b"][f"{shp} faktor={faktor}"] = info["shapes_b"].get(f"{shp} faktor={faktor}", 0) + 1
        info["spacing_faktoren"][pid] = faktor
        info["faelle_gesamt"] += 1
        print(pid, "->", shp, "factor", faktor, flush=True)
    # Warning + note: cohort 1 is interpolated in z from 4.0 to 1.0
    info["warnung"] = ("Cohort 1: z-spacing 4.0 mm -> 1.0 mm, i.e. 3x "
                       "linear interpolation INVENTS intermediate slices that did not "
                       "exist in any measurement. This is exactly why the A/B comparison: "
                       "B must show that the network learns bleeds, "
                       "not cohort geometry.")
    json.dump(info, open(f"{ROOT}/dev/gitter_b_info.json", "w"), indent=1, ensure_ascii=False)
    print("DONE:", info["faelle_b"], "cases resampled")

if __name__ == "__main__":
    main()
