#!/usr/bin/env python3
"""dev/preproc_nobias (Claude 2026-09-19): input image WITHOUT bias-field correction and without vessel
inpainting, to measure what FSL FAST -B contributes. Same brain mask and same formula as dev/preproc_roh:
image = 1 - raw / max(raw) inside the mask, 0 outside. Then:
  GITTER_C_QUELLE=dev/preproc_nobias GITTER_C_ZIEL=dev/gitter_h python gitter_c_bauen.py
"""
import os, glob, shutil
import numpy as np, nibabel as nib
ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VALDO = os.environ.get("CMB_VALDO", os.path.join(ROOT, "data", "valdo2021", "Task2"))
for d in sorted(glob.glob(f"{ROOT}/dev/preproc/sub-*")):
    sid = os.path.basename(d); z = f"{ROOT}/dev/preproc_nobias/{sid}"; os.makedirs(z, exist_ok=True)
    if os.path.exists(f"{z}/{sid}_image.nii.gz"):
        continue
    ref = nib.load(f"{d}/{sid}_image.nii.gz"); brain = np.asarray(ref.dataobj) > 0
    raw = np.asarray(nib.load(f"{VALDO}/{sid}/{sid}_space-T2S_desc-masked_T2S.nii.gz").dataobj, dtype=np.float32)
    assert raw.shape == brain.shape, sid
    image = np.where(brain, 1 - raw / raw.max(), 0).astype(np.float32)
    nib.save(nib.Nifti1Image(image, ref.affine, ref.header), f"{z}/{sid}_image.nii.gz")
    shutil.copy2(f"{d}/{sid}_label.nii.gz", f"{z}/{sid}_label.nii.gz")
    print(sid, flush=True)
