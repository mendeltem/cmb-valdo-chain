"""``cmb predict``: microbleeds on one T2*-GRE scan (or a folder of scans), as simple as HD-BET.

    cmb predict -i scan.nii.gz -o out/                      # one scan
    cmb predict -i scans/ -o out/                           # every *.nii / *.nii.gz in the folder
    cmb predict scan.nii.gz --frst frst.nii.gz              # positional form, own FRST map (-frst works too)
    cmb predict -i scan.nii.gz -m my_finetuned_model/       # a model you fine-tuned yourself (e.g. on your SWI scans)

What it runs (each step is skipped when you pass its result):
  1. brain mask      --brain-mask; else image > 0 when the scan is already skull-stripped (>= 20 % exact zeros);
                     else HD-BET when ``hd-bet`` is on the PATH; else a rough intensity mask (WARNING printed)
  2. bias correction FSL FAST -B exactly as for the paper (``fast -t 2 -n 3 -B -b --nopve``) when FSL is found;
                     else SimpleITK N4 (WARNING: not the paper's step); --no-bias-correction if your scan is corrected
  3. inversion       1 - I / max(I) inside the brain (microbleeds bright), as in the paper
  4. grid            0.5 x 0.5 x 1.0 mm + brain crop, with the paper's grid builder (code/gitter_c_bauen.py)
  5. FRST channel    computed on the grid as in the paper; --frst FILE instead resamples your map (WARNING printed)
  6. stage 1         the VALDO-72 U-Net (two seeds averaged; downloaded once, SHA-256 checked)
  7. post-processing the paper's fixed cell: probability > 0.3, 26-connected components of >= 2 mm3 and <= 2100 voxels
  8. CSF rule        optional (--csf-rule): drop components whose majority SynthSeg label is CSF; needs FreeSurfer's
                     mri_synthseg (run on the bias-corrected scan); without FreeSurfer it is skipped with a WARNING
  9. back to the scan: probability map and labelled mask resampled to the ORIGINAL scan grid

Outputs per scan in -o: <name>_cmb_mask.nii.gz (labelled components 1..n), <name>_cmb_prob.nii.gz (probability 0..1),
<name>_cmb.csv (one row per CMB) and one summary line on screen.

Stage 2 (the candidate classifier) is not part of this command: no classifier trained on public data alone exists as a
file (PAPER_VERSION.md). The result is the paper's stage-1 chain with the fixed post-processing (the "n = 0" arm).

The public model saw T2*-GRE only (VALDO). Other sequences such as SWI need local fine-tuning first
(``cmb.training`` / scripts/paper/phase2_finetune/run_block.py, see README) and then ``-m <your model folder>``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import numpy as np

from cmb.paths import ROOT

CSF_LABELS = (4, 5, 14, 15, 24, 43, 44)
GRID_MM = (0.5, 0.5, 1.0)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def stem_of(path: str) -> str:
    b = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return b


def scans_in(path: str):
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*.nii.gz")) + glob.glob(os.path.join(path, "*.nii")))
        if not found:
            raise SystemExit(f"no .nii/.nii.gz files in {path}")
        return found
    if not os.path.exists(path):
        raise SystemExit(f"input not found: {path}")
    return [path]


# ----------------------------------------------------------------------------------------------- preprocessing steps
def brain_mask(img: np.ndarray, ni, given: Optional[str], work: str, log) -> np.ndarray:
    import nibabel as nib
    from scipy import ndimage
    if given:
        m = np.asarray(nib.load(given).dataobj) > 0
        if m.shape != img.shape:
            raise SystemExit(f"brain mask {m.shape} does not match the scan {img.shape}")
        log("brain mask: given")
        return m
    zeros = float((img == 0).mean())
    if zeros >= 0.20:
        log(f"brain mask: image > 0 (scan looks skull-stripped, {zeros:.0%} zero voxels)")
        return ndimage.binary_fill_holes(img > 0)
    if shutil.which("hd-bet"):
        src = os.path.join(work, "hdbet_in.nii.gz"); out = os.path.join(work, "hdbet_out.nii.gz")
        nib.save(nib.Nifti1Image(img.astype(np.float32), ni.affine, ni.header), src)
        r = subprocess.run(["hd-bet", "-i", src, "-o", out], capture_output=True, text=True)
        masks = glob.glob(os.path.join(work, "hdbet_out*mask*.nii.gz"))
        if r.returncode == 0 and masks:
            log("brain mask: HD-BET")
            return np.asarray(nib.load(masks[0]).dataobj) > 0
        warn(f"HD-BET failed (rc {r.returncode}); falling back to an intensity mask")
    warn("no brain mask given, scan not skull-stripped and HD-BET not found: using a rough intensity mask "
         "(Otsu + largest component). Pass --brain-mask for reliable results.")
    from skimage.filters import threshold_otsu
    m = img > threshold_otsu(img[np.isfinite(img)])
    lab, n = ndimage.label(m)
    if n:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3))


def find_fast() -> Optional[str]:
    for c in (os.path.join(os.environ.get("FSLDIR", ""), "bin", "fast"), shutil.which("fast")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def bias_correct(masked: np.ndarray, ni, mode: str, work: str, log) -> np.ndarray:
    import nibabel as nib
    if mode == "none":
        log("bias correction: skipped (--no-bias-correction)")
        return masked
    fast = find_fast() if mode in ("auto", "fsl") else None
    if fast:
        src = os.path.join(work, "masked.nii.gz"); stem = os.path.join(work, "fast")
        nib.save(nib.Nifti1Image(masked.astype(np.float32), ni.affine, ni.header), src)
        env = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
        r = subprocess.run([fast, "-t", "2", "-n", "3", "-B", "-b", "--nopve", "-o", stem, src],
                           env=env, capture_output=True, text=True)      # the paper's call (code/preprocess.py)
        if r.returncode == 0 and os.path.exists(stem + "_restore.nii.gz"):
            log("bias correction: FSL FAST -B (as in the paper)")
            return np.asarray(nib.load(stem + "_restore.nii.gz").dataobj, dtype=np.float32)
        warn(f"FSL FAST failed (rc {r.returncode}): {r.stderr[-200:]}")
    if mode == "fsl":
        raise SystemExit("--bias fsl: FSL FAST not available")
    import SimpleITK as sitk
    warn("FSL FAST not found: bias correction with SimpleITK N4 instead (not the paper's step; results may differ)")
    im = sitk.GetImageFromArray(np.ascontiguousarray(masked.transpose(2, 1, 0)).astype(np.float32))
    mk = sitk.GetImageFromArray(np.ascontiguousarray((masked > 0).transpose(2, 1, 0)).astype(np.uint8))
    out = sitk.N4BiasFieldCorrectionImageFilter().Execute(im, mk)
    return np.where(masked > 0, sitk.GetArrayFromImage(out).transpose(2, 1, 0), 0).astype(np.float32)


def invert(restore: np.ndarray) -> np.ndarray:
    """code/preproc_roh_bauen.py: 1 - restore / max(restore) inside the brain (restore > 0, holes filled), else 0."""
    from scipy import ndimage
    brain = ndimage.binary_fill_holes(restore > 0)
    return np.where(brain, 1 - restore / restore.max(), 0).astype(np.float32)


PROFILES = ("own", "valdo")


def source_volumes(restore: np.ndarray, brain: np.ndarray, label: np.ndarray, ni, profile: str = "own"):
    """The image the grid builder starts from (inverted, 0 outside the brain), its label and its affine/header, exactly
    as the paper made them:

    * ``own``  -- the paper's pipeline for its own (in-house) cohorts, ``cmb.transfer.build_private_grids.prepare_source``:
      bias-corrected image, brain mask and label reoriented to RAS+ (``nibabel.as_closest_canonical``), brain = mask AND
      image > 0, ``1 - I / max(I in the brain)``, voxels at 0 inside the brain set to 1e-6, saved without the scan header.
    * ``valdo`` -- the VALDO grid ``dev/gitter_d`` (``code/preproc_roh_bauen.py``): no reorientation, brain = image > 0
      with holes filled, ``1 - I / max(I)``, saved with the scan header.
    Returns (inverted, label uint8, affine, header or None)."""
    import nibabel as nib
    if profile == "valdo":
        return invert(restore), (label > 0).astype(np.uint8), ni.affine, ni.header
    if profile != "own":
        raise SystemExit(f"unknown profile {profile!r} (own | valdo)")
    image_nii = nib.as_closest_canonical(nib.Nifti1Image(restore.astype(np.float32), ni.affine, ni.header))
    canon = lambda a: np.asarray(nib.as_closest_canonical(nib.Nifti1Image(a.astype(np.uint8), ni.affine)).dataobj) > 0
    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    b, lab = canon(brain), canon(label)
    b &= image > 0
    inverted = np.where(b, 1.0 - image / image[b].max(), 0.0).astype(np.float32)
    inverted[b & (inverted <= 0)] = 1e-6                                     # 0 is reserved for "outside the brain"
    return inverted, lab.astype(np.uint8), image_nii.affine, None


def build_grid(case: str, inverted: np.ndarray, affine, header, work: str, label: np.ndarray = None, dst: str = None) -> dict:
    """The paper's grid builder (code/gitter_c_bauen.py ``fall``) on a one-case source folder. header None = saved
    without a header, as prepare_source of the own-cohort pipeline does."""
    import nibabel as nib
    src, dst = os.path.join(work, "source"), dst or os.path.join(work, "grid")
    os.makedirs(os.path.join(src, case), exist_ok=True)
    label = np.zeros(inverted.shape, np.uint8) if label is None else label.astype(np.uint8)
    if header is None:
        nib.save(nib.Nifti1Image(inverted, affine), f"{src}/{case}/{case}_image.nii.gz")
        nib.save(nib.Nifti1Image(label, affine), f"{src}/{case}/{case}_label.nii.gz")
    else:
        hdr = header.copy(); hdr.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(inverted, affine, hdr), f"{src}/{case}/{case}_image.nii.gz")
        hdr8 = header.copy(); hdr8.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(label, affine, hdr8), f"{src}/{case}/{case}_label.nii.gz")
    sys.path.insert(0, os.path.join(ROOT, "code"))
    keep = os.environ.get("CUDA_VISIBLE_DEVICES")
    import gitter_c_bauen as builder                  # its import hides the GPU for its own worker processes ...
    if keep is None:                                  # ... restore the setting for this process (CUDA is not initialised yet)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = keep
    builder.QUELLE, builder.ZIEL = src, dst
    info = builder.fall((case, 0, False))
    if "fehler" in info:
        raise SystemExit(f"grid builder failed: {info['fehler']}")
    info["dir"] = os.path.join(dst, case)
    shutil.rmtree(os.path.join(src, case), ignore_errors=True)
    return info


def own_frst_on_grid(frst_path: str, ni, info: dict, grid_shape, profile: str = "own") -> np.ndarray:
    """Resample a user-supplied FRST map (on the scan grid) with the grid builder's index transform (reoriented to RAS+
    first for the own profile, like the image)."""
    import nibabel as nib
    from scipy import ndimage
    fni = nib.load(frst_path)
    if fni.shape[:3] != ni.shape[:3]:
        raise SystemExit(f"--frst {fni.shape} does not match the scan {ni.shape}")
    if profile == "own":
        fni = nib.as_closest_canonical(nib.Nifti1Image(np.asarray(fni.dataobj, dtype=np.float32), ni.affine))
    fr = np.asarray(fni.dataobj, dtype=np.float32)
    s = np.sqrt((fni.affine[:3, :3] ** 2).sum(0))
    r = np.array(GRID_MM) / s
    off = np.array(info["crop_offset_quellvoxel"], float)
    return ndimage.affine_transform(fr, r, offset=off, output_shape=tuple(grid_shape), order=1, mode="nearest",
                                    output=np.float32)


# ----------------------------------------------------------------------------------------------- model
def load_models(model_dir: str, device: str):
    os.environ["CV5_DEV"] = device
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import torch
    import cv5
    import turnier  # noqa: F401  -- registers the networks in cv5
    cv5.DEV = device
    files = sorted(glob.glob(os.path.join(model_dir, "*", "modell.pt"))) or sorted(glob.glob(os.path.join(model_dir, "modell.pt")))
    if not files:
        raise SystemExit(f"no <member>/modell.pt (with meta.json next to it) in {model_dir}")
    models = []
    for p in files:
        meta = json.load(open(os.path.join(os.path.dirname(p), "meta.json")))
        model, nch = cv5.neues_modell(meta["params"])
        model.load_state_dict(torch.load(p, map_location=device))
        models.append((model.to(device).eval(), nch))
    return models


def predict_grid(models, case: str, image: np.ndarray, frst: np.ndarray, affine) -> np.ndarray:
    """Mean of cv5.proba_fall over the members, quantised to uint8 / 255 like the stored maps of the paper."""
    import cv5
    turn = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
    cv5.FA = dict(recs={case: dict(img=turn(image), frs=turn(frst), fold=0, affine=affine)})
    p = np.mean([cv5.proba_fall(m, case, nch=nch)[0] for m, nch in models], axis=0)
    p8 = np.round(p * 255).astype(np.uint8)
    return np.ascontiguousarray(p8.transpose(2, 1, 0)).astype(np.float32) / 255.0      # (x, y, z)


# ----------------------------------------------------------------------------------------------- CSF rule
def synthseg_on_grid(restore_path: str, grid_img, work: str, log) -> Optional[np.ndarray]:
    import nibabel as nib
    import nibabel.processing
    fs = os.path.join(os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer"), "bin", "mri_synthseg")
    exe = fs if os.path.exists(fs) else shutil.which("mri_synthseg")
    if not exe:
        warn("--csf-rule: mri_synthseg (FreeSurfer) not found -> CSF rule skipped. Expect more false positives in "
             "sulcal/ventricular CSF (on VALDO the rule removed 98 of 176 stage-1 false positives).")
        return None
    out = os.path.join(work, "synthseg.nii.gz")
    r = subprocess.run([exe, "--i", restore_path, "--o", out, "--robust", "--cpu", "--threads", "4"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        warn(f"mri_synthseg failed (rc {r.returncode}) -> CSF rule skipped")
        return None
    log("CSF rule: SynthSeg on the bias-corrected scan")
    seg = nibabel.processing.resample_from_to(nib.load(out), grid_img, order=0)
    return np.asarray(seg.dataobj).astype(np.int32)


# ----------------------------------------------------------------------------------------------- one scan
def run_one(scan: str, out_dir: str, model_dir: str, models, frst: Optional[str] = None, mask: Optional[str] = None,
            bias: str = "auto", csf: bool = False, keep_work: bool = False, log=print, cell: dict = None,
            profile: str = "own") -> dict:
    import nibabel as nib
    import nibabel.processing  # noqa: F401
    from scipy import ndimage
    from cmb.transfer.evaluate import THRESHOLD, MIN_MM3, MAX_VOXELS, VOXEL_MM3
    from cmb.analysis.missed_lesions import majority_label
    t0 = time.time()
    name = stem_of(scan)
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f".cmb_{name}_", dir=out_dir)
    try:
        ni = nib.load(scan)
        img = np.asarray(ni.dataobj, dtype=np.float32)
        if img.ndim == 4 and img.shape[3] == 1:
            img = img[..., 0]
        if img.ndim != 3:
            raise SystemExit(f"{scan}: need a 3D scan, got shape {img.shape}")
        img = np.nan_to_num(img)
        brain = brain_mask(img, ni, mask, work, log)
        restore = bias_correct(np.where(brain, img, 0).astype(np.float32), ni, bias, work, log)
        inverted, _, src_affine, src_header = source_volumes(restore, brain, np.zeros(img.shape, np.uint8), ni, profile)
        case = "case0"
        info = build_grid(case, inverted, src_affine, src_header, work)
        g_img = nib.load(f"{info['dir']}/{case}_image.nii.gz")
        image = np.asarray(g_img.dataobj, dtype=np.float32)
        if frst:
            warn("--frst: using your FRST map resampled to the grid; the model was trained on the FRST this code "
                 "computes on the grid (code/frst.py, radii 1-3 mm). Different scaling changes the result.")
            fr = own_frst_on_grid(frst, ni, info, image.shape, profile)
        else:
            fr = np.asarray(nib.load(f"{info['dir']}/{case}_frst.nii.gz").dataobj, dtype=np.float32)
        prob = predict_grid(models, case, image, fr, g_img.affine)
        cell = cell or dict(threshold=THRESHOLD, min_mm3=MIN_MM3, max_voxels=MAX_VOXELS)   # the paper's fixed cell
        comp, nc = ndimage.label(prob > cell["threshold"], structure=np.ones((3, 3, 3), bool))   # = evaluate.post_process
        size = np.bincount(comp.ravel(), minlength=nc + 1)
        ok = (size * VOXEL_MM3 >= cell["min_mm3"]) & (size <= cell["max_voxels"]); ok[0] = False
        keep = ok[comp]
        labels, n = ndimage.label(keep, structure=np.ones((3, 3, 3), bool))
        n_csf = 0
        if csf:
            rpath = os.path.join(work, "restore.nii.gz")
            nib.save(nib.Nifti1Image(restore, ni.affine), rpath)
            seg = synthseg_on_grid(rpath, g_img, work, log)
            if seg is not None:
                for c, box in enumerate(ndimage.find_objects(labels), start=1):
                    if box is not None and majority_label(labels[box] == c, seg[box]) in CSF_LABELS:
                        labels[labels == c] = 0; n_csf += 1
                labels, n = ndimage.label(labels > 0, structure=np.ones((3, 3, 3), bool))
        # rows on the grid (0.5 x 0.5 x 1 mm): volume, diameter of the largest single-slice cross-section
        rows = []
        for c, box in enumerate(ndimage.find_objects(labels), start=1):
            m = labels[box] == c
            vox = int(m.sum()); vol = vox * float(np.prod(GRID_MM))
            area = float(m.sum(axis=(0, 1)).max()) * GRID_MM[0] * GRID_MM[1]          # largest axial cross-section, mm2
            diam = 2.0 * np.sqrt(area / np.pi)
            cg = np.array([v + s.start for v, s in zip(ndimage.center_of_mass(m), box)])
            world = (g_img.affine @ np.r_[cg, 1.0])[:3]
            native = (np.linalg.inv(ni.affine) @ np.r_[world, 1.0])[:3]
            rows.append(dict(id=c, x_vox=round(float(native[0]), 1), y_vox=round(float(native[1]), 1),
                             z_vox=round(float(native[2]), 1), x_mm=round(float(world[0]), 2),
                             y_mm=round(float(world[1]), 2), z_mm=round(float(world[2]), 2),
                             volume_mm3=round(vol, 2), diameter_mm=round(diam, 2),
                             size_class="<2 mm" if diam < 2 else ("2-5 mm" if diam <= 5 else ">5 mm"),
                             stage1_max_prob=round(float(prob[box][m].max()), 3), stage2_score=""))
        # back to the original scan grid
        ref = nib.Nifti1Image(np.zeros(ni.shape[:3], np.float32), ni.affine)
        p_nat = nib.processing.resample_from_to(nib.Nifti1Image(prob, g_img.affine), ref, order=1, cval=0.0)
        l_nat = nib.processing.resample_from_to(nib.Nifti1Image(labels.astype(np.int16), g_img.affine), ref, order=0, cval=0)
        hdr = ni.header.copy(); hdr.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(np.clip(np.asarray(p_nat.dataobj, np.float32), 0, 1), ni.affine, hdr),
                 os.path.join(out_dir, f"{name}_cmb_prob.nii.gz"))
        hdr.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(np.asarray(l_nat.dataobj).astype(np.int16), ni.affine, hdr),
                 os.path.join(out_dir, f"{name}_cmb_mask.nii.gz"))
        with open(os.path.join(out_dir, f"{name}_cmb.csv"), "w", newline="") as f:
            fields = ["id", "x_vox", "y_vox", "z_vox", "x_mm", "y_mm", "z_mm", "volume_mm3", "diameter_mm", "size_class",
                      "stage1_max_prob", "stage2_score"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        classes = {k: sum(r["size_class"] == k for r in rows) for k in ("<2 mm", "2-5 mm", ">5 mm")}
        print(f"{name}: {len(rows)} CMB ({classes['<2 mm']} <2 mm, {classes['2-5 mm']} 2-5 mm, {classes['>5 mm']} >5 mm)"
              + (f", {n_csf} removed by the CSF rule" if csf else "") + f"  [{time.time() - t0:.0f} s] -> {out_dir}",
              flush=True)
        return dict(name=name, n=len(rows), classes=classes, cell=cell)
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------------------------- command line
def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("input", nargs="?", help="scan (.nii/.nii.gz) or folder of scans (same as -i)")
    ap.add_argument("-i", "--input", dest="input_opt", help="scan or folder of scans")
    ap.add_argument("-o", "--output", default=None, help="output folder (default: next to the scan, <name>_cmb/ ... "
                    "for a folder: <folder>_cmb/)")
    ap.add_argument("--frst", "-frst", default=None, help="your own FRST map on the scan grid (default: computed)")
    ap.add_argument("--brain-mask", "-b", default=None, help="brain mask on the scan grid (default: see steps)")
    ap.add_argument("-m", "--model", default=None,
                    help="model folder (<member>/modell.pt + meta.json, as written by cmb.training.full_data or "
                         "scripts/paper/phase2_finetune/run_block.py). Default: the public VALDO T2* model")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--bias", default="auto", choices=["auto", "fsl", "n4"], help="bias correction method")
    ap.add_argument("--no-bias-correction", action="store_true", help="the scan is already bias-corrected")
    ap.add_argument("--csf-rule", action="store_true", help="drop components in CSF (needs FreeSurfer mri_synthseg)")
    ap.add_argument("--keep-work", action="store_true", help="keep the intermediate grid files")
    ap.add_argument("--profile", choices=PROFILES, default="own",
                    help="how the scan becomes the model's input: own = the paper's pipeline for its own cohorts (RAS+ "
                         "reorientation, brain = mask and image > 0; default); valdo = exactly the VALDO grid dev/gitter_d")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cmb predict", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    a = ap.parse_args(argv)
    run(a)


def run(a) -> None:
    src = a.input_opt or a.input
    if not src:
        raise SystemExit("give a scan or folder: cmb predict -i scan.nii.gz [-o out/]")
    scans = scans_in(src)
    if a.frst and len(scans) > 1:
        raise SystemExit("--frst works with a single scan only")
    if a.brain_mask and len(scans) > 1:
        raise SystemExit("--brain-mask works with a single scan only")
    out = a.output or (src.rstrip("/") + "_cmb" if os.path.isdir(src) else os.path.join(os.path.dirname(os.path.abspath(src)),
                                                                                        stem_of(src) + "_cmb"))
    import torch
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if a.model:
        model_dir = a.model
        print(f"model: own model {model_dir} (the sequence you trained it on)", flush=True)
    else:
        from cmb.pretrained import get_model, TAG
        model_dir = get_model("valdo_t2")
        print(f"model: VALDO T2* (T2*-GRE scans only), cmb-valdo-chain {TAG}, stage 1 + fixed cell 0.3 / 2 mm3; "
              f"for other sequences fine-tune locally and use -m", flush=True)
    print(f"device: {device}; {len(scans)} scan(s) -> {out}", flush=True)
    models = load_models(model_dir, device)
    from cmb.commands import model_cell
    cell = model_cell(model_dir)
    print(f"post-processing: probability > {cell['threshold']}, components >= {cell['min_mm3']} mm3 and "
          f"<= {cell['max_voxels']} voxels" + ("" if a.model else " (the paper's fixed cell, chosen on VALDO)"), flush=True)
    bias = "none" if a.no_bias_correction else a.bias
    total, results = 0, []
    for s in scans:
        r = run_one(s, out, model_dir, models, frst=a.frst, mask=a.brain_mask, bias=bias, csf=a.csf_rule,
                    keep_work=a.keep_work, log=lambda m: print("  " + m, flush=True), cell=cell, profile=a.profile)
        total += r["n"]
        r.update(output=out, csv=os.path.join(out, f"{r['name']}_cmb.csv"))
        results.append(r)
    if len(scans) > 1:
        print(f"{len(scans)} scans, {total} CMB in total -> {out}")
    return results


if __name__ == "__main__":
    main()
