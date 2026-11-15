"""cmb.maskfix.build_v2: rebuild the private model grids (0.5x0.5x1 mm) from the RAW scans
using the FILLED brain masks. Only change vs cmb.transfer.build_private_grids:

  * brain mask = <hirn>_filled.nii.gz (enclosed holes filled, 3D + axial 2D);
  * the line  `brain &= image > 0`  is DROPPED. In the old build it cut every voxel the
    SWI phase mask drives to 0 back out of the brain, so the microbleeds became holes
    (image 0) instead of the brightest value. With the filled mask and no image>0 cut,
    a raw 0 inside the brain becomes 1 - 0/max = 1.0, the brightest value, as intended.

Everything else is byte-for-byte the same code path: biascorr source, resampling to
0.5 x 0.5 x 1.0 mm, crop to the brain bounding box + 8 voxels, FRST recomputed, labels
resampled with the nearest neighbour. The grid code is gitter_c_bauen.fall, which is
reused by pointing its QUELLE/ZIEL at the v2 folders. Output is written to NEW folders
<private>/gitter/{swi,t2star}_v2 and the manifests manifest_{swi_mix,t2star}_v2.json carry
the SAME ids and folds as the originals (cmb.analysis.fold_freeze must pass).

    python -m cmb.maskfix.build_v2 --modality swi
    python -m cmb.maskfix.build_v2 --modality t2star
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from multiprocessing import Pool
import nibabel as nib, numpy as np

from cmb.paths import PRIVATE, INHOUSE_LABEL, CASES, ID_PREFIX
sys.path.insert(0, __import__("cmb.paths").paths.ROOT + "/code")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
import gitter_c_bauen as builder
from cmb.maskfix.fill import filled_path


def prepare_source_v2(entry: dict, modality: str, source_dir: str) -> str:
    case = f"{ID_PREFIX[modality]}-{entry['id']}"
    folder = f"{source_dir}/{case}"
    if os.path.exists(f"{folder}/{case}_label.nii.gz"):
        return case
    os.makedirs(folder, exist_ok=True)
    corrected = glob.glob(os.path.dirname(entry["bild"]).replace(
        f"/{entry['id']}/anat", f"/derivatives/biascorr/{entry['id']}/anat") + "/*biascorr*.nii.gz")
    image_nii = nib.as_closest_canonical(nib.load(corrected[0] if corrected else entry["bild"]))
    squeeze = lambda a: a[..., 0] if a.ndim == 4 else a
    image = squeeze(np.asarray(image_nii.dataobj, dtype=np.float32))
    filled = filled_path(entry["hirn"])
    if not os.path.exists(filled):
        filled = os.path.join(f"{PRIVATE}/maskfix/filled/{modality}", os.path.basename(filled))
    brain = squeeze(np.asarray(nib.as_closest_canonical(nib.load(filled)).dataobj)) > 0
    label = squeeze(np.asarray(nib.as_closest_canonical(nib.load(entry["maske"])).dataobj)) > 0
    assert image.shape == brain.shape == label.shape, (case, image.shape, brain.shape, brain.shape, label.shape)
    # NO `brain &= image > 0` here: a raw 0 inside the (filled) brain is the brightest value.
    inverted = np.where(brain, 1.0 - image / image[brain].max(), 0.0).astype(np.float32)
    inverted[brain & (inverted <= 0)] = 1e-6
    nib.save(nib.Nifti1Image(inverted, image_nii.affine), f"{folder}/{case}_image.nii.gz")
    nib.save(nib.Nifti1Image(label.astype(np.uint8), image_nii.affine), f"{folder}/{case}_label.nii.gz")
    return case


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["t2star", "swi"], required=True)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--faelle", nargs="*", default=None)
    args = ap.parse_args()
    entries = json.load(open(CASES))[args.modality]["faelle"]
    if args.faelle:
        entries = [e for e in entries if e["id"] in args.faelle]
    source_dir, target_dir = f"{PRIVATE}/gitter/quelle_{args.modality}_v2", f"{PRIVATE}/gitter/{args.modality}_v2"
    os.makedirs(target_dir, exist_ok=True)
    builder.QUELLE, builder.ZIEL = source_dir, target_dir
    jobs = [(prepare_source_v2(e, args.modality, source_dir), e["fold"], True) for e in entries]
    with Pool(args.processes, maxtasksperchild=2) as pool:
        infos = pool.map(builder.fall, jobs)
    failed = [i["id"] for i in infos if "fehler" in i]
    changed = [i["id"] for i in infos if "fehler" not in i and not i["komponenten_gleich"]]
    outside = [i["id"] for i in infos if "fehler" not in i and not i["alle_laesionen_im_crop"]]
    manifest = [dict(id=c, fold=f, dir=f"{target_dir}/{c}", cohort=f"{INHOUSE_LABEL}-{args.modality}",
                     n_cmb=next(i.get("n_cmb") for i in infos if i["id"] == c))
                for c, f, _ in jobs if c not in failed]
    mpath = f"{PRIVATE}/gitter/manifest_{args.modality}_v2.json"
    if args.modality == "swi" and args.faelle is None:
        # swi_mix: the seven T2* twins sit in their T2* fold, exactly like the original mix manifest
        old = json.load(open(f"{PRIVATE}/gitter/manifest_swi_mix.json"))
        folds = {e["id"]: e["fold"] for e in old}
        for e in manifest:
            e["fold"] = folds[e["id"]]
        mpath = f"{PRIVATE}/gitter/manifest_swi_mix_v2.json"
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"{args.modality}: {len(manifest)} of {len(jobs)} on the v2 grid | failed {len(failed)} | "
          f"lesion count changed {len(changed)} | outside crop {len(outside)} | manifest {mpath}")


if __name__ == "__main__":
    main()
