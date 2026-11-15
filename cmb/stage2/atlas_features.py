"""Arena 2: venous-atlas values at every stage-2 candidate (2026-10-25, research round 3 R5).

Why: the main false positives are vein cross-sections, and a candidate crop alone cannot always tell. Where large veins run is
known at population level: Bernier 2018 (Human Brain Mapping, DOI 10.1002/hbm.24337) averaged the vessel extraction of 41 healthy
3 T SWI scans in MNI space (mean_Ved_swi_Thresh = per-voxel vein frequency 0..1; GPL-3.0; <vein-atlas>).
This is information from OTHER people's scans, not from the candidate's pixels. No CMB paper has used a vein atlas (searched
2026-10-25); in aneurysm detection an arterial location prior cut false positives sharply (Di Noto 2022). The values become three
stage-2 features (module 'atlas' of cmb.stage2.trackset), never a hard rule.

Three steps:
  register  per case: SynthMorph (FreeSurfer 8.2, contrast-agnostic, needs no T1) registers the MNI152 brain template (moving) to
            the case image on the model grid (fixed); the same transform carries the atlas onto the case grid:
            <out>/<case>_vein.nii.gz. Existing outputs are skipped. CPU ~2.5 min per case (12 threads), -g for the GPU.
            The atlas is first resampled (world coordinates, linear) onto the TEMPLATE grid: 'mri_synthmorph apply' on an image
            with a different grid than the moving image gave garbage in the probe of 2026-10-25 (template-derived scan on our
            0.5 x 0.5 x 1 mm grid, inverted: correlation with the true atlas -0.02 / Dice 0.02 directly, 0.917 / 0.78 after
            resampling; the template itself was registered well, correlation 0.961).
  features  copies a candidate folder and adds per candidate: atlas_vein_p (value at the centre voxel), atlas_vein_max3mm (maximum
            within 3 mm), atlas_vein_dist_mm (distance to the nearest voxel >= --level, capped at 20 mm); patches.npy is linked,
            not copied.
  auc       the 30-minute check BEFORE any GPU run: per cohort the AUC of each value for 'false candidate' (a vein should score
            higher on false candidates). Sums only.

    python -m cmb.stage2.atlas_features register --grid dev/gitter_d --cases-from ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 \
        --out dev/atlas_vein
    python -m cmb.stage2.atlas_features features --candidates ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 \
        --maps "dev/atlas_vein/{id}_vein.nii.gz" --out ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas
    python -m cmb.stage2.atlas_features auc --candidates valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40-atlas t2star=...
Private cohorts: --manifest instead of --grid, and private --out folders.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Dict, List

import numpy as np

from cmb.paths import ROOT
from cmb.paths import FREESURFER_HOME as FS
from cmb.paths import MNI_TEMPLATE as TEMPLATE
from cmb.paths import VEIN_ATLAS as ATLAS
GRID_MM_XYZ = np.array([0.5, 0.5, 1.0])
FIELDS = ("atlas_vein_p", "atlas_vein_max3mm", "atlas_vein_dist_mm")


def absolute(p: str) -> str:
    return p if os.path.isabs(p) else f"{ROOT}/{p}"


def case_images(grid, manifests, cases: List[str]) -> Dict[str, str]:
    if manifests:
        dirs = {e["id"]: e["dir"] for m in manifests for e in json.load(open(absolute(m)))}
        return {c: f"{dirs[c]}/{c}_image.nii.gz" for c in cases if c in dirs}
    return {c: f"{absolute(grid)}/{c}/{c}_image.nii.gz" for c in cases}


def atlas_on_template(atlas_path: str, template_path: str, out_path: str) -> str:
    """Resample the atlas onto the template grid via world coordinates (linear); cached."""
    import nibabel as nib
    from scipy import ndimage
    if not os.path.exists(out_path):
        atl, tpl = nib.load(atlas_path), nib.load(template_path)
        m = np.linalg.inv(atl.affine) @ tpl.affine
        data = ndimage.affine_transform(np.asarray(atl.dataobj, np.float32), m[:3, :3], m[:3, 3], output_shape=tpl.shape, order=1)
        nib.save(nib.Nifti1Image(data.astype(np.float32), tpl.affine), out_path)
    return out_path


def register(args) -> None:
    cases = sorted(json.load(open(f"{absolute(args.cases_from)}/candidates.json"))["cases"])
    images = case_images(args.grid, args.manifest, cases)
    out = absolute(args.out); os.makedirs(out, exist_ok=True)
    env = dict(os.environ, FREESURFER_HOME=FS, PATH=f"{FS}/bin:" + os.environ.get("PATH", ""))
    if not args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    atlas = atlas_on_template(args.atlas, TEMPLATE, f"{out}/atlas_on_template_grid.nii.gz") if not args.dry_run else args.atlas
    done = 0
    for case, image in images.items():
        vein = f"{out}/{case}_vein.nii.gz"
        if os.path.exists(vein):
            continue
        trans = f"{out}/{case}_mni2case.nii.gz"
        cmds = [[f"{FS}/bin/mri_synthmorph", "register", "-m", "joint", "-t", trans, "-j", str(args.threads)] + (["-g"] if args.gpu else [])
                + [TEMPLATE, image],
                [f"{FS}/bin/mri_synthmorph", "apply", "-m", "linear", trans, atlas, vein]]
        for cmd in cmds:
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
        done += 1
    print(f"{done} cases registered, {len(images) - done} already present; maps in {out}")


def features_for_case(vein: np.ndarray, centres_xyz: np.ndarray, level: float) -> np.ndarray:
    """vein: atlas on the case grid (x, y, z); -> (n, 3) values per candidate centre voxel (x, y, z)."""
    from scipy import ndimage
    r = np.ceil(3.0 / GRID_MM_XYZ).astype(int)
    xx, yy, zz = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    ball = (xx * GRID_MM_XYZ[0]) ** 2 + (yy * GRID_MM_XYZ[1]) ** 2 + (zz * GRID_MM_XYZ[2]) ** 2 <= 9.0
    vmax = ndimage.maximum_filter(vein, footprint=ball, mode="constant")
    dist = ndimage.distance_transform_edt(vein < level, sampling=GRID_MM_XYZ) if (vein >= level).any() else np.full(vein.shape, 20.0)
    out = np.zeros((len(centres_xyz), 3), np.float32)
    for i, (x, y, z) in enumerate(centres_xyz):
        x, y, z = (int(np.clip(v, 0, s - 1)) for v, s in zip((x, y, z), vein.shape))
        out[i] = (vein[x, y, z], vmax[x, y, z], min(float(dist[x, y, z]), 20.0))
    return out


def features(args) -> None:
    import nibabel as nib
    src = absolute(args.candidates); out = absolute(args.out)
    if os.path.abspath(src) == os.path.abspath(out):
        raise SystemExit("--out must differ from --candidates")
    info = json.load(open(f"{src}/candidates.json"))
    by_case: Dict[str, List[int]] = {}
    for i, r in enumerate(info["candidates"]):
        by_case.setdefault(r["case"], []).append(i)
    missing = []
    for case, idx in by_case.items():
        path = absolute(args.maps.format(id=case))
        if not os.path.exists(path):
            missing.append(case); continue
        vein = np.asarray(nib.load(path).dataobj, np.float32)
        vals = features_for_case(vein, np.array([info["candidates"][i]["centre_voxel"] for i in idx]), args.level)
        for i, v in zip(idx, vals):
            info["candidates"][i].update({k: round(float(x), 4) for k, x in zip(FIELDS, v)})
    if missing:
        raise SystemExit(f"{len(missing)} cases without atlas map (run 'register' first), e.g. {missing[:3]}")
    os.makedirs(out, exist_ok=True)
    info["atlas"] = dict(atlas=args.atlas, maps=args.maps, level=args.level)
    json.dump(info, open(f"{out}/candidates.json", "w"))
    link = f"{out}/patches.npy"
    if not os.path.exists(link):
        os.symlink(os.path.realpath(f"{src}/patches.npy"), link)
    print(f"{len(info['candidates'])} candidates in {len(by_case)} cases with atlas values -> {out}")


def auc(args) -> None:
    def roc_auc(score, y):
        pos, neg = score[y == 1], score[y == 0]
        if not len(pos) or not len(neg):
            return float("nan")
        order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
        ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        allv = np.concatenate([pos, neg])                   # average ranks for ties
        for v in np.unique(allv):
            m = allv == v
            ranks[m] = ranks[m].mean()
        return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    for spec in args.candidates:
        name, folder = spec.split("=", 1)
        rows = json.load(open(f"{absolute(folder)}/candidates.json"))["candidates"]
        false = np.array([0 if r["touched"] else 1 for r in rows])
        parts = []
        for f in FIELDS:
            v = np.array([r[f] for r in rows], float)
            parts.append(f"{f} {roc_auc(-v if f.endswith('dist_mm') else v, false):.3f}")
        print(f"{name}: {len(rows)} candidates ({int(false.sum())} false); AUC for 'false candidate': " + ", ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register"); r.add_argument("--grid"); r.add_argument("--manifest", nargs="+")
    r.add_argument("--cases-from", required=True, help="candidate folder whose cases.json lists the cases to register")
    r.add_argument("--out", required=True); r.add_argument("--atlas", default=ATLAS); r.add_argument("--threads", type=int, default=8)
    r.add_argument("--gpu", action="store_true"); r.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("features"); f.add_argument("--candidates", required=True); f.add_argument("--maps", required=True)
    f.add_argument("--out", required=True); f.add_argument("--level", type=float, default=0.5); f.add_argument("--atlas", default=ATLAS)
    a = sub.add_parser("auc"); a.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "register" and bool(args.grid) == bool(args.manifest):
        raise SystemExit("give exactly one of --grid and --manifest")
    dict(register=register, features=features, auc=auc)[args.command](args)


if __name__ == "__main__":
    main()
