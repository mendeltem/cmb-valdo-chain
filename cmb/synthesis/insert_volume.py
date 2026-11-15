"""Arena 2: insert physics-rendered microbleeds and vein mimics into WHOLE host volumes (2026-10-25).

Why whole volumes (research round 3, R3): the stage-2 classifier sees three channels -- image, FRST, stage-1 probability. A
rendered object changes only the image; FRST and the U-Net probability must be recomputed on the modified volume, or the classifier
learns 'channels disagree = synthetic'. This script writes modified copies of the host cases in the grid layout
(<out>/<id>/<id>_{image,frst,label}.nii.gz + manifest_synthetic.json with the HOST'S fold); then
  1. cmb.transfer.zero_shot --own-fold --manifest <out>/manifest_synthetic.json  (each copy only through the U-Net of its own
     fold, which never saw the host -- out-of-fold like the real candidates),
  2. cmb.stage2.candidates on those maps (same --half / --keep-csf as the pool),
  3. cmb.synthesis.realism_probe (must pass before any training),
  4. cmb.stage2.pooled ... --extra NAME=<synthetic candidates>: training-only candidates of the same case ids and folds, so
     nothing leaks across folds (R3: fit priors and pick hosts inside the training folds only).

Objects (cmb.synthesis.dipole_render): microbleeds (positives, labelled with their APPARENT extent A < 0.6, like a rater who
outlines the dark blob) and veins at random angles to B0 (negatives, no label). NO calcifications: on magnitude they are the same
image as a microbleed of equal |moment| (R3) and would contradict the positive labels.
Placement: inside the brain, not in CSF (SynthSeg labels, if given), >= 8 mm from reference lesions and from other insertions;
region shares lobar 0.70 / deep 0.12 / infratentorial 0.18 after AGES-Reykjavik (R9). Veins are finite segments: their signal
loss fades out over the last 4 mm of the render box, so no box edge shows. Acquisition geometry per cohort (B0 T, TE ms, slice
thickness mm, in-plane mm) is given on the command line -- cohort-level metadata, not per case.

    python -m cmb.synthesis.insert_volume --manifest <hosts manifest with id, dir, fold> --synthseg "<pattern {id}>" \
        --b0 3 --te 20 --slice-mm 5 --inplane-mm 0.5 --bleeds 3 --veins 3 --seed 0 --out <private>/synthetic/<cohort>-s0
Private hosts -> private --out. Prints totals only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from cmb.synthesis import dipole_render as R

from cmb.paths import ROOT
sys.path.insert(0, f"{ROOT}/code")
GRID_MM_XYZ = np.array([0.5, 0.5, 1.0])
CSF = {4, 5, 14, 15, 24, 43, 44}
REGIONS = {"lobar": ({2, 3, 41, 42}, 0.70), "deep": ({10, 11, 12, 13, 49, 50, 51, 52, 28, 60}, 0.12),
           "infratentorial": ({7, 8, 16, 46, 47}, 0.18)}
HALF_BLEED = np.array([16, 16, 8])        # render box half size in voxels (x, y, z) = 8 x 8 x 8 mm
HALF_VEIN = np.array([28, 28, 14])        # 14 x 14 x 14 mm
FADE_MM = 4.0
APPARENT = 0.6


def choose_sites(brain: np.ndarray, seg: Optional[np.ndarray], forbidden: np.ndarray, n: int, rng, min_mm: float = 8.0) -> List[np.ndarray]:
    """n voxel centres (x, y, z) inside the brain (eroded 2 mm), outside CSF, >= min_mm from `forbidden` and from each other."""
    inner = ndimage.binary_erosion(brain, iterations=2)
    if seg is not None:
        inner &= ~np.isin(seg, list(CSF))
    far = ndimage.distance_transform_edt(~forbidden, sampling=GRID_MM_XYZ) >= min_mm if forbidden.any() else np.ones_like(brain)
    allowed = inner & far
    pools = []
    if seg is not None:
        for labels, share in REGIONS.values():
            idx = np.argwhere(allowed & np.isin(seg, list(labels)))
            if len(idx):
                pools.append((idx, share))
    if not pools:
        pools = [(np.argwhere(allowed), 1.0)]
    shares = np.array([s for _, s in pools]); shares = shares / shares.sum()
    sites: List[np.ndarray] = []
    for _ in range(50 * n):
        if len(sites) == n:
            break
        idx, _ = pools[rng.choice(len(pools), p=shares)]
        c = idx[rng.randint(len(idx))]
        if all(np.linalg.norm((c - s) * GRID_MM_XYZ) >= min_mm for s in sites):
            sites.append(c)
    return sites


def fade_window(shape_zyx, voxel_zyx, fade_mm: float) -> np.ndarray:
    """1 inside, falling smoothly (cosine) to 0 over the last fade_mm towards every face of the box."""
    w = np.ones(shape_zyx, np.float32)
    for axis, (n, v) in enumerate(zip(shape_zyx, voxel_zyx)):
        d = np.minimum(np.arange(n), np.arange(n)[::-1]) * v
        f = np.clip(d / fade_mm, 0, 1); f = 0.5 - 0.5 * np.cos(np.pi * f)
        shape = [1, 1, 1]; shape[axis] = n
        w *= f.reshape(shape)
    return w


def render_into(image: np.ndarray, centre_xyz: np.ndarray, obj_kind: str, rng, geometry: Dict) -> Dict:
    """Render one object at centre_xyz into `image` (x, y, z, inverted grid image) in place; -> log entry incl. apparent mask box."""
    half = HALF_BLEED if obj_kind == "bleed" else HALF_VEIN
    lo = np.maximum(centre_xyz - half, 0); hi = np.minimum(centre_xyz + half + 1, image.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    host = np.ascontiguousarray(image[box].transpose(2, 1, 0))                    # (z, y, x)
    voxel_zyx = GRID_MM_XYZ[::-1]
    centre_mm = ((centre_xyz - lo)[::-1] + 0.5) * voxel_zyx + rng.uniform(-0.25, 0.25, 3) * voxel_zyx
    obj = R.sample_object("bleed" if obj_kind == "bleed" else "vein", rng, tuple(centre_mm))
    b_hat = R.tilted_b0(rng)
    acq = (geometry["slice_mm"], geometry["inplane_mm"], geometry["inplane_mm"])
    A = R.render(host.shape, voxel_zyx, obj, geometry["b0"], geometry["te"], acq_voxel_mm=acq, b_hat=b_hat,
                 fine_mm=(0.25, 0.125, 0.125) if obj_kind == "bleed" else (0.25, 0.1, 0.1),
                 slab_offset_mm=rng.uniform(0, 1, 3) * np.array(acq))
    if obj_kind == "vein":
        A = 1.0 - (1.0 - A) * fade_window(A.shape, voxel_zyx, FADE_MM)
    entry = dict(kind=obj_kind, centre_voxel=[int(v) for v in centre_xyz], radius_mm=round(float(obj["radius"]), 3),
                 chi_ppm=round(float(obj["chi"]), 3), b_hat=[round(float(v), 3) for v in b_hat],
                 min_attenuation=round(float(A.min()), 3), deficit_mm3=round(R.deficit_volume(A, voxel_zyx), 2))
    if obj_kind == "vein":
        entry["axis"] = [round(float(v), 3) for v in obj["axis"]]
    apparent = np.zeros(image.shape, bool)
    if obj_kind == "bleed":
        lab, _ = ndimage.label(A < APPARENT)
        c = tuple(np.clip(np.round(centre_mm / voxel_zyx - 0.5).astype(int), 0, np.array(A.shape) - 1))
        k = lab[c] if lab[c] else (np.bincount(lab.ravel())[1:].argmax() + 1 if lab.max() else 0)
        if k:
            apparent[box] = (lab == k).transpose(2, 1, 0)
    entry["apparent_voxels"] = int(apparent.sum())
    # a microbleed too faint to outline is NOT written into the image: it would be an unlabelled real-looking bleed (label noise)
    if obj_kind == "vein" or entry["apparent_voxels"]:
        image[box] = R.compose_inverted(host, A, rng).transpose(2, 1, 0)
        entry["written"] = True
    else:
        entry["written"] = False
    return entry, apparent


def main() -> None:
    import nibabel as nib
    import frst as frstmod
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", nargs="+", required=True, help="host manifests: list of {id, dir, fold}")
    parser.add_argument("--synthseg", default=None, help="pattern with {id}: SynthSeg on the grid (placement); none = brain only")
    parser.add_argument("--b0", type=float, required=True); parser.add_argument("--te", type=float, required=True)
    parser.add_argument("--slice-mm", type=float, required=True); parser.add_argument("--inplane-mm", type=float, required=True)
    parser.add_argument("--bleeds", type=int, default=3); parser.add_argument("--veins", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    geometry = dict(b0=args.b0, te=args.te, slice_mm=args.slice_mm, inplane_mm=args.inplane_mm)
    hosts = [e for m in args.manifest for e in json.load(open(m))]
    os.makedirs(args.out, exist_ok=True)
    manifest, log, n_b, n_v, n_faint = [], {}, 0, 0, 0
    for e in hosts:
        cid, d = e["id"], e["dir"]
        rng = np.random.RandomState(args.seed * 100003 + sum(map(ord, cid)))
        ni = nib.load(f"{d}/{cid}_image.nii.gz")
        image = np.asarray(ni.dataobj, np.float32).copy()
        label = np.asarray(nib.load(f"{d}/{cid}_label.nii.gz").dataobj) > 0
        seg = None
        if args.synthseg:
            from cmb.analysis.fp_location import load_synthseg, synthseg_path
            seg = load_synthseg(synthseg_path(args.synthseg, cid), ni)
        brain = image > 0
        sites = choose_sites(brain, seg, label, args.bleeds + args.veins, rng)
        kinds = ["bleed"] * args.bleeds + ["vein"] * args.veins
        new_label = label.copy(); entries = []
        for kind, site in zip(kinds, sites):
            entry, apparent = render_into(image, site, kind, rng, geometry)
            if kind == "bleed":
                if entry["apparent_voxels"] == 0:
                    n_faint += 1; entry["labelled"] = False                 # not written either (see render_into)
                else:
                    new_label |= apparent; entry["labelled"] = True; n_b += 1
            else:
                n_v += 1
            entries.append(entry)
        image[~brain] = 0.0
        fr, _ = frstmod.frst_volumen(image, frstmod.hirnmaske(image))
        od = f"{args.out}/{cid}"; os.makedirs(od, exist_ok=True)
        for name, arr, dtype in (("image", image, np.float32), ("frst", fr, np.float32), ("label", new_label, np.uint8)):
            img = nib.Nifti1Image(arr.astype(dtype), ni.affine, ni.header); img.set_data_dtype(dtype)
            nib.save(img, f"{od}/{cid}_{name}.nii.gz")
        manifest.append(dict(id=cid, dir=od, fold=e["fold"], host_dir=d, synthetic=True))
        log[cid] = entries
    json.dump(manifest, open(f"{args.out}/manifest_synthetic.json", "w"), indent=1)
    json.dump(dict(geometry=geometry, bleeds=args.bleeds, veins=args.veins, seed=args.seed, objects=log),
              open(f"{args.out}/insert_log.json", "w"), indent=1)
    print(f"{len(hosts)} host cases: {n_b} labelled bleeds, {n_faint} too faint (skipped), {n_v} veins -> {args.out}")


if __name__ == "__main__":
    main()
