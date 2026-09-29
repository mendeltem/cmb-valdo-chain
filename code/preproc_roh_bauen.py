#!/usr/bin/env python3
"""dev/preproc_roh (Claude 19.09.2026): dasselbe Eingangsbild wie dev/preproc, aber OHNE die
Gefaess-Uebermalung der MicrobleedNet-Kette. Anlass (protokoll.md 19.09.): die Uebermalung trifft
109 von 236 Referenz-CMB, 27 ganz, 17 sind danach unsichtbar.

Bild = 1 - fast_restore / max(fast_restore) innerhalb der Hirnmaske von dev/preproc (Bild > 0), sonst 0.
Geprueft: an unbemalten Voxeln ist das identisch mit dev/preproc (Median-Abweichung 0.0000).
Label wird kopiert. Danach:
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
