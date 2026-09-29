#!/usr/bin/env python3
"""Builds the inputs for microbleednet from faelle.json:

  eingaben/<mod>/bilder/<id>_img.nii.gz    = raw image * brain mask (float32, header+grid of the raw image)
  eingaben/<mod>/labels/<id>_label.nii.gz  = CMB mask > 0 (uint8)

Grid check (nifti-pruefen): mask and brain mask must sit on exactly the same
grid as the raw image (same matrix; corner offset < 0.25 * smallest voxel
edge). Every file is read (not just the header: np.asarray).
Parallel over cases (multiprocessing), since it is purely CPU/IO.
"""
import json, os, sys, time
import numpy as np
import nibabel as nib
from multiprocessing import Pool

B = os.path.dirname(os.path.abspath(__file__))
PY = "/home/uchralt/miniconda3/envs/dl/bin/python"

def _pruefe_gitter(bild, andres):
    """Returns (ok, message): matrix match + corner offset in mm against tolerance."""
    if bild.shape[:3] != andres.shape[:3]:
        return False, f"matrix {andres.shape[:3]} != {bild.shape[:3]}"
    nx, ny, nz = bild.shape[:3]
    c = np.array([[0, 0, 0, 1], [nx, 0, 0, 1], [0, ny, 0, 1], [0, 0, nz, 1],
                  [nx, ny, 0, 1], [nx, 0, nz, 1], [0, ny, nz, 1], [nx, ny, nz, 1]], float).T
    off = float(np.abs((bild.affine @ c)[:3] - (andres.affine @ c)[:3]).max())
    tol = 0.25 * min(bild.header.get_zooms()[:3])
    return off <= tol, f"corner offset {off:.3f} mm (tol {tol:.3f} mm)"

def _baue_einen(arg):
    mod, c = arg
    bild_p, maske_p, hirn_p = c["bild"], c["maske"], c["hirn"]
    bid = nib.load(bild_p);  bild = np.asarray(bid.dataobj)
    mas = nib.load(maske_p); maske = np.asarray(mas.dataobj)
    hir = nib.load(hirn_p);  hirn = np.asarray(hir.dataobj)
    ok_m, meld_m = _pruefe_gitter(bid, mas)
    ok_h, meld_h = _pruefe_gitter(bid, hir)
    if not (ok_m and ok_h):
        return dict(id=c["id"], mod=mod, ok=False, meldung=f"mask: {meld_m}; brain: {meld_h}")
    img = (bild * (hirn > 0).astype(float)).astype(np.float32)
    lab = (maske > 0).astype(np.uint8)
    outdir = os.path.join(B, "eingaben", mod)
    os.makedirs(f"{outdir}/bilder", exist_ok=True)
    os.makedirs(f"{outdir}/labels", exist_ok=True)
    nib.save(nib.Nifti1Image(img, bid.affine, bid.header), f"{outdir}/bilder/{c['id']}_img.nii.gz")
    nib.save(nib.Nifti1Image(lab, bid.affine, bid.header), f"{outdir}/labels/{c['id']}_label.nii.gz")
    z = bid.header.get_zooms()[:3]
    return dict(id=c["id"], mod=mod, ok=True, shape=tuple(bid.shape[:3]), zooms=tuple(round(float(v), 3) for v in z),
                n_cmb=int(lab.sum()))

def main():
    t0 = time.time()
    faelle = json.load(open(f"{B}/faelle.json"))
    jobs = [(mod, c) for mod in ("swi", "t2star") for c in faelle[mod]["faelle"]]
    with Pool(12) as p:
        results = p.map(_baue_einen, jobs)
    fehler = [r for r in results if not r["ok"]]
    for mod in ("swi", "t2star"):
        rs = [r for r in results if r["mod"] == mod]
        print(f"{mod}: {len(rs)}/{len(faelle[mod]['faelle'])} cases built in "
              f"{int(time.time()-t0)} s, grid {rs[0]['shape']}, voxel {rs[0]['zooms']} mm, "
              f"CMB sum {sum(r['n_cmb'] for r in rs)}")
    for r in fehler:
        print("ERROR", r["mod"], r["id"], r["meldung"])
    if fehler:
        sys.exit(1)

if __name__ == "__main__":
    main()
