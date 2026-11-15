"""Run-stopping input checks of the command line (``cmb preprocess``, ``cmb predict``, the manifest commands).

Each check stops the run with a message that names the files and what to do, instead of letting a wrong input give
plausible but wrong numbers. They encode the errors found after paper-v3-2026-11 (CHANGES.md):

* a mask or label on another voxel grid or in another orientation than its scan (the mirrored x axis of LAS/LPS
  scans): ``same_grid``
* a brain mask or a CMB label that lies where the scan has no data: ``mask_inside_image``, ``label_inside_image``
* two cases with the same id (every scan got the same name when the scan suffix was split wrongly): ``unique_ids``
* labelled scans left out of the lesion metric (CMB-free scans dropped from the F1): ``all_labelled_scored``

On valid input none of them changes anything: they only read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

# Grid tolerance of the paper's own input check (scripts/prep_bids/eingaben_bauen.py): the corners of the voxel grid
# may differ by at most a quarter of the smallest voxel edge (header rounding of real scans stays far below it; a
# grid shifted or flipped by one voxel or more does not).
CORNER_TOLERANCE = 0.25
# A brain mask with more than this fraction of its voxels on zero-valued scan voxels does not belong to the scan (a
# skull-stripped scan is zero outside the brain, a scan that is not stripped is non-zero almost everywhere).
MAX_MASK_ON_EMPTY = 0.5


def _corners(shape: tuple[int, ...]) -> np.ndarray:
    nx, ny, nz = shape[:3]
    return np.array([[x, y, z, 1] for x in (0, nx) for y in (0, ny) for z in (0, nz)], float).T


def _axcodes(affine: np.ndarray) -> str:
    import nibabel as nib

    return "".join(str(c) for c in nib.aff2axcodes(affine))


def same_grid(scan, other, scan_path: str, other_path: str) -> None:
    """Stop unless ``other`` (mask, label, FRST map) is on the voxel grid of ``scan``: same matrix, same orientation
    code (e.g. RAS), voxel corners within CORNER_TOLERANCE of the smallest voxel edge. ``scan``/``other`` are
    nibabel images."""
    s_shape, o_shape = tuple(scan.shape[:3]), tuple(other.shape[:3])
    if s_shape != o_shape:
        raise SystemExit(
            f"{other_path}: matrix {o_shape} does not match the scan {s_shape} ({scan_path}); "
            "masks and labels must be on the scan's own voxel grid"
        )
    a, b = np.asarray(scan.affine, float), np.asarray(other.affine, float)
    ca, cb = _axcodes(a), _axcodes(b)
    if ca != cb:
        raise SystemExit(
            f"{other_path}: orientation {cb} but the scan is {ca} ({scan_path}); the voxels would be "
            "mirrored against the scan -- write the mask in the scan's orientation (same affine)"
        )
    offset = float(np.abs((a @ _corners(s_shape))[:3] - (b @ _corners(s_shape))[:3]).max())
    tol = CORNER_TOLERANCE * float(np.sqrt((a[:3, :3] ** 2).sum(0)).min())
    if offset > tol:
        raise SystemExit(
            f"{other_path}: voxel grid differs from the scan ({scan_path}) by {offset:.3f} mm at the "
            f"corners (tolerance {tol:.3f} mm); the affines must agree"
        )


def mask_inside_image(image: np.ndarray, mask: np.ndarray, mask_path: str) -> None:
    """Stop when the brain mask is empty or mostly lies on zero-valued (or non-finite) scan voxels."""
    m = np.asarray(mask) > 0
    if m.ndim == 4 and m.shape[3] == 1:
        m = m[..., 0]
    n = int(m.sum())
    if n == 0:
        raise SystemExit(f"{mask_path}: the brain mask is empty")
    img = np.asarray(image)
    empty = float((~np.isfinite(img[m]) | (img[m] == 0)).mean())
    if empty > MAX_MASK_ON_EMPTY:
        raise SystemExit(
            f"{mask_path}: {empty:.0%} of the brain mask lies where the scan is 0 -- the mask does not "
            "belong to this scan (another case, another grid?)"
        )


def label_inside_image(image: np.ndarray, label: np.ndarray, label_path: str) -> None:
    """Stop when a labelled CMB (26-connected component) lies entirely where the scan has no data (0 or non-finite)."""
    from cmb.nifti import components

    lab = np.asarray(label) > 0
    if not lab.any():
        return
    img = np.asarray(image)
    has_data = np.isfinite(img) & (img != 0)
    comp, n = components(lab, np.ones((3, 3, 3), bool))
    seen = np.zeros(n + 1, bool)
    seen[np.unique(comp[lab & has_data])] = True
    outside = int((~seen[1:]).sum())
    if outside:
        raise SystemExit(
            f"{label_path}: {outside} of {n} labelled CMB lie where the scan is 0 -- the label does not "
            "belong to this scan (another case, mirrored?)"
        )


def unique_ids(entries: Iterable[dict], where: str) -> None:
    """Stop when a manifest has an empty or a repeated case id."""
    ids = [str(e.get("id", "")) for e in entries]
    if any(not i for i in ids):
        raise SystemExit(f"{where}: a case has an empty id")
    twice = sorted(i for i, k in Counter(ids).items() if k > 1)
    if twice:
        raise SystemExit(
            f"{where}: case id(s) {', '.join(twice[:5])} occur more than once -- every case needs its own id"
        )


def all_labelled_scored(n_scored: int, entries: Iterable[dict]) -> None:
    """Stop when the lesion metric did not score every labelled scan of the manifest (CMB-free ones included)."""
    n_labelled = sum(bool(e.get("labelled", True)) for e in entries)
    if n_scored != n_labelled:
        raise SystemExit(
            f"scored {n_scored} scans but the manifest has {n_labelled} labelled scans: every labelled "
            "scan must be scored, CMB-free ones included"
        )
