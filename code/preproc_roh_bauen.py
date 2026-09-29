#!/usr/bin/env python3
"""dev/preproc_roh (Claude 19 Sep 2026): the same input image as dev/preproc, but WITHOUT the
vessel inpainting of the MicrobleedNet chain. Reason (protokoll.md 19 Sep): the inpainting hits
109 of 236 reference CMBs, 27 completely, 17 are invisible afterward.

Image = 1 - fast_restore / max(fast_restore) inside the brain mask of dev/preproc (image > 0), else 0.
Checked: at unpainted voxels this is identical to dev/preproc (median deviation 0.0000).
Label is copied. Afterward:
  GITTER_C_QUELLE=dev/preproc_roh GITTER_C_ZIEL=dev/gitter_d python gitter_c_bauen.py
"""
import os, sys, glob, shutil
import numpy as np, nibabel as nib
ROOT = "/home/uchralt/data/work/valdo-t2s"
for d in sorted(glob.glob(f"{ROOT}/dev/preproc/sub-*")):
    sid = os.path.basename(d); z = f"{ROOT}/dev/preproc_roh/{sid}"; os.makedirs(z, exist_ok=True)
    if os.path.exists(f"{z}/{sid}_image.nii.gz"):
        continue
    ni = nib.load(f"{d}/{sid}_image.nii.gz"); hirn = np.asarray(ni.dataobj) > 0
    res = np.asarray(nib.load(f"{d}/fast_restore.nii.gz").dataobj, dtype=np.float32)
    bild = np.where(hirn, 1 - res / res.max(), 0).astype(np.float32)
    nib.save(nib.Nifti1Image(bild, ni.affine, ni.header), f"{z}/{sid}_image.nii.gz")
    shutil.copy2(f"{d}/{sid}_label.nii.gz", f"{z}/{sid}_label.nii.gz")
    print(sid, flush=True)
