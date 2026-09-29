"""Bring the private cohorts (T2*-GRE and SWI) onto the same training grid as VALDO.

Identical treatment to the VALDO data of the current best recipe (grid ``dev/gitter_d``):
  bias-corrected image -> brain mask -> inverted scale ``1 - image / max`` inside the mask
  -> NO vessel inpainting (it cost 0.12 F1 on VALDO) -> resampling to 0.5 x 0.5 x 1.0 mm,
  cropped to the brain -> radial-symmetry map computed on that grid.
The resampling and the radial-symmetry code are reused from ``code/gitter_c_bauen.py`` so that both
cohorts go through literally the same functions.

Differences that cannot be avoided and must be kept in mind when reading results:
  * brain masks come from HD-BET here (tight), VALDO ships its own (loose in two cohorts);
  * images are reoriented to RAS+ first; slice thickness is 5 mm (T2*) / 2 mm (SWI), so the
    through-plane direction is interpolated even more strongly than in VALDO's 4 mm cohort.

Private data: everything is written below ``mb-arena/`` and never leaves the machine.

    python -m cmb.transfer.build_private_grids --modality t2star
    python -m cmb.transfer.build_private_grids --modality swi
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

PRIVATE = "/home/uchralt/data/work/mb-arena"
sys.path.insert(0, "/home/uchralt/data/work/valdo-t2s/code")


def prepare_source(entry: dict, modality: str, source_dir: str) -> str:
    """Write ``<id>_image.nii.gz`` (inverted, masked) and ``<id>_label.nii.gz`` on the native grid."""
    case = f"{'c2s' if modality == 't2star' else 'csw'}-{entry['id']}"          # unique across all cohorts
    folder = f"{source_dir}/{case}"
    if os.path.exists(f"{folder}/{case}_label.nii.gz"):
        return case
    os.makedirs(folder, exist_ok=True)
    corrected = glob.glob(os.path.dirname(entry["bild"]).replace(
        f"/{entry['id']}/anat", f"/derivatives/biascorr/{entry['id']}/anat") + "/*biascorr*.nii.gz")
    image_nii = nib.as_closest_canonical(nib.load(corrected[0] if corrected else entry["bild"]))
    squeeze = lambda a: a[..., 0] if a.ndim == 4 else a
    image = squeeze(np.asarray(image_nii.dataobj, dtype=np.float32))
    brain = squeeze(np.asarray(nib.as_closest_canonical(nib.load(entry["hirn"])).dataobj)) > 0
    label = squeeze(np.asarray(nib.as_closest_canonical(nib.load(entry["maske"])).dataobj)) > 0
    assert image.shape == brain.shape == label.shape, (case, image.shape, brain.shape, label.shape)
    brain &= image > 0
    inverted = np.where(brain, 1.0 - image / image[brain].max(), 0.0).astype(np.float32)
    inverted[brain & (inverted <= 0)] = 1e-6                                     # 0 is reserved for "outside the brain"
    nib.save(nib.Nifti1Image(inverted, image_nii.affine), f"{folder}/{case}_image.nii.gz")
    nib.save(nib.Nifti1Image(label.astype(np.uint8), image_nii.affine), f"{folder}/{case}_label.nii.gz")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=["t2star", "swi"], required=True)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    entries = json.load(open(f"{PRIVATE}/faelle.json"))[args.modality]["faelle"]
    source_dir, target_dir = f"{PRIVATE}/gitter/quelle_{args.modality}", f"{PRIVATE}/gitter/{args.modality}"
    os.makedirs(target_dir, exist_ok=True)

    jobs = [(prepare_source(e, args.modality, source_dir), e["fold"], True) for e in entries]
    import gitter_c_bauen as builder                         # legacy module; paths are module globals
    builder.QUELLE, builder.ZIEL = source_dir, target_dir
    with Pool(args.processes, maxtasksperchild=2) as pool:   # fork: the children inherit the two globals
        infos = pool.map(builder.fall, jobs)

    failed = [i["id"] for i in infos if "fehler" in i]
    changed = [i["id"] for i in infos if "fehler" not in i and not i["komponenten_gleich"]]
    outside = [i["id"] for i in infos if "fehler" not in i and not i["alle_laesionen_im_crop"]]
    manifest = [dict(id=c, fold=f, dir=f"{target_dir}/{c}", cohort=f"catalina-{args.modality}",
                     n_cmb=next(i.get("n_cmb") for i in infos if i["id"] == c)) for c, f, _ in jobs if c not in failed]
    json.dump(manifest, open(f"{PRIVATE}/gitter/manifest_{args.modality}.json", "w"), indent=1)
    print(f"{args.modality}: {len(manifest)} of {len(jobs)} cases on the grid | failed {len(failed)} | "
          f"lesion count changed by resampling {len(changed)} | lesions outside the crop {len(outside)}")


if __name__ == "__main__":
    main()
