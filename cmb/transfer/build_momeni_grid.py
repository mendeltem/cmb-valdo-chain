"""Momeni/CSIRO SWI cohort (doi 10.25919/aegy-ny12; licence: non-commercial, attribution, NO redistribution) onto the
training grid, as the first INDEPENDENT public test of the SWI model (ARBEIT.md 5ae, arm 3; 2026-09-28).

Source: ~/data/extern/momeni-csiro-cmb/PublicDataShare_2020/rCMB_DefiniteSubject/*.nii.gz -- 57 SWI, skull-stripped,
bias-corrected, histogram-matched by the authors, 0.94 x 0.94 x 1.75 mm, LPS; expert CMB LOCATIONS (no masks) in
rCMBInformationInfo.xlsx, parsed to rCMB_locations.json as 1-based (x, y, z) voxel indices (checked 2026-09-28: at these
voxels the image is 5.5 SD darker than the 13-voxel neighbourhood; every other axis order / index base is ~0).

Treatment identical to VALDO/catalina: brain = image > 0, inverted scale 1 - I / max inside the brain, then the VALDO
grid builder (code/gitter_c_bauen.py: 0.5 x 0.5 x 1.0 mm, brain crop, FRST) via its environment switches. The label
is a SPHERE of radius 1.5 mm around every location -- a stand-in for the missing masks, good enough for the
centre-of-mass metric (cmb.analysis.subject_level_f1) and, with the touch rule, an approximation for the project metric.
Ids are ``mom-<file stem>``; folds are assigned 0..4 round-robin (only a folder name for the prediction tools).

    python -m cmb.transfer.build_momeni_grid [--processes 8]
Writes <extern>/gitter/<id>/... and <extern>/gitter/manifest_momeni.json. Public data, but the licence forbids
redistribution: nothing from here goes into any repository.
"""
from __future__ import annotations
import argparse, json, os, sys
import nibabel as nib, numpy as np
from scipy import ndimage

EXTERN = "/home/uchralt/data/extern/momeni-csiro-cmb"
ROOT = "/home/uchralt/data/work/valdo-t2s"
SRC = f"{EXTERN}/PublicDataShare_2020/rCMB_DefiniteSubject"
PY = "/home/uchralt/miniconda3/envs/dl/bin/python"


def prepare(entry, source_dir):
    stem = entry["file"].replace(".nii.gz", ""); case = "mom-" + stem.split("_")[0] + "-" + stem.split("_")[1]   # mom-122-T1
    folder = f"{source_dir}/{case}"; os.makedirs(folder, exist_ok=True)
    ni = nib.load(f"{SRC}/{entry['file']}"); img = np.asarray(ni.dataobj, dtype=np.float32)
    brain = img > 0
    inv = np.where(brain, 1.0 - img / img[brain].max(), 0.0).astype(np.float32); inv[brain & (inv <= 0)] = 1e-6
    sp = np.array(ni.header.get_zooms()[:3], float); lab = np.zeros(img.shape, np.uint8)
    rad = np.ceil(1.5 / sp).astype(int)
    for p in entry["points"]:
        c = np.array(p) - 1                                    # 1-based -> 0-based, order x, y, z as stored
        lo = np.maximum(c - rad, 0); hi = np.minimum(c + rad + 1, img.shape)
        g = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d2 = sum(((g[i] - c[i]) * sp[i]) ** 2 for i in range(3))
        lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= (d2 <= 1.5 ** 2)
    assert int(ndimage.label(lab, structure=np.ones((3, 3, 3)))[1]) == len(entry["points"]), case
    nib.save(nib.Nifti1Image(inv, ni.affine), f"{folder}/{case}_image.nii.gz")
    nib.save(nib.Nifti1Image(lab, ni.affine), f"{folder}/{case}_label.nii.gz")
    return case, len(entry["points"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processes", type=int, default=8); a = ap.parse_args()
    source_dir, grid = f"{EXTERN}/quelle", f"{EXTERN}/gitter"
    entries = json.load(open(f"{EXTERN}/rCMB_locations.json"))
    cases = [prepare(e, source_dir) for e in entries]
    os.makedirs(grid, exist_ok=True)
    sys.path.insert(0, f"{ROOT}/code")
    import gitter_c_bauen as builder                          # same functions as VALDO / catalina (build_private_grids)
    builder.QUELLE, builder.ZIEL = source_dir, grid
    from multiprocessing import Pool
    jobs = [(case, k % 5, True) for k, (case, _) in enumerate(cases)]
    with Pool(a.processes, maxtasksperchild=2) as pool:
        infos = pool.map(builder.fall, jobs)
    failed = [i["id"] for i in infos if "fehler" in i]
    assert not failed, failed
    manifest = []
    for k, (case, n) in enumerate(cases):
        lab = np.asarray(nib.load(f"{grid}/{case}/{case}_label.nii.gz").dataobj) > 0
        n_grid = int(ndimage.label(lab, structure=np.ones((3, 3, 3)))[1])
        manifest.append(dict(id=case, fold=k % 5, dir=f"{grid}/{case}", cohort="momeni-swi", n_cmb=n_grid, n_points=n))
    print("lesion count changed by resampling:", [m["id"] for m in manifest if m["n_cmb"] != m["n_points"]])
    json.dump(manifest, open(f"{grid}/manifest_momeni.json", "w"), indent=1)
    print(f"{len(manifest)} cases, {sum(m['n_cmb'] for m in manifest)} lesions on the grid ({sum(m['n_points'] for m in manifest)} points) -> {grid}/manifest_momeni.json")


if __name__ == "__main__":
    main()
