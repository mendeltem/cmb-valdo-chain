"""cmb.maskfix: brain masks with holes (SWI phase mask drives microbleeds to 0, the mask was
cut at image > 0, so the darkest spots became holes) are filled BEFORE masking/inversion.

Hole definition = holes_of in qc/build_model_input_qc.py: 3D fill plus 2D fill of every
axial slice; an open notch touching the border is NOT a hole and is not filled.

    python -m pytest -q cmb/tests/test_maskfix.py
    python -m cmb.maskfix.fill --modality swi        (writes <mask>_filled.nii.gz next to the original)
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

from cmb.paths import PRIVATE, CASES


def holes_of(brain: np.ndarray) -> np.ndarray:
    """Voxels enclosed by the brain mask but not in it: 3D fill plus axial 2D fill."""
    brain = brain.astype(bool)
    filled = ndimage.binary_fill_holes(brain)
    filled |= np.stack([ndimage.binary_fill_holes(brain[..., k]) for k in range(brain.shape[2])], -1)
    return filled & ~brain


def fill_mask(brain: np.ndarray) -> np.ndarray:
    """Brain mask with all enclosed holes filled. Open notches (touching the image border)
    are left untouched."""
    return brain.astype(bool) | holes_of(brain)


def filled_path(mask_path: str) -> str:
    """<mask>_filled.nii.gz next to the original; if the original is not writable
    (sourcedata is read-only) the caller passes out_dir and the file lands in
    out_dir/<basename> with _filled inserted."""
    base, ext = os.path.splitext(mask_path)
    if mask_path.endswith(".nii.gz"):
        return mask_path[:-len(".nii.gz")] + "_filled.nii.gz"
    return f"{base}_filled{ext}"


def fill_case(mask_path: str, out_path: str = None, force: bool = False, out_dir: str = None) -> str:
    """Fill one brain mask: new file <mask>_filled.nii.gz (same header, dtype). The
    original is NEVER overwritten. If the original's folder is not writable, write to
    out_dir/<name>_filled.nii.gz instead. Idempotent. Returns the output path."""
    if out_path is None:
        out_path = filled_path(mask_path)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(out_path))
    img = nib.load(mask_path)
    data = np.asarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    brain = data > 0
    if os.path.exists(out_path) and not force:
        return out_path                                          # idempotent
    out = fill_mask(brain).astype(data.dtype)                    # same dtype as the original mask
    nib.save(nib.Nifti1Image(out, img.affine, img.header), out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["swi", "t2star"], required=True)
    ap.add_argument("--faelle", nargs="*", default=None, help="sub-ids only (default: all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=None, help="write here if the original's folder is not writable")
    args = ap.parse_args()
    entries = json.load(open(CASES))[args.modality]["faelle"]
    if args.faelle:
        entries = [e for e in entries if e["id"] in args.faelle]
    n = 0
    for e in entries:
        try:
            out = fill_case(e["hirn"], force=args.force)
        except PermissionError:
            out = fill_case(e["hirn"], force=args.force, out_dir=args.out_dir or f"{PRIVATE}/maskfix/filled/{args.modality}")
        n += 1
        print(f"{e['id']}: {out}", flush=True)
    print(f"done {n} masks")


if __name__ == "__main__":
    main()
