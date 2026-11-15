"""wk5 (31.10.2026): MARS location classes for CMB, pre-specified and not tuned.

Definition (Gregoire et al. 2009, Neurology 73:1759-66, doi:10.1212/WNL.0b013e3181c34a7d),
operationalised on the SynthSeg maps this project already uses (labels as in
cmb.stage2.candidates):

  infratentorial  brainstem (16) or cerebellum (7, 8, 46, 47); sub-class brainstem / cerebellum
  deep            thalamus, caudate, putamen, pallidum, accumbens, ventral DC (10-13, 26, 28, 49-52, 58, 60),
                  AND cerebral white matter (2, 41) whose distance to the lateral ventricles (4, 43, 5, 44)
                  is SMALLER than its distance to the cortex (3, 42)
  lobar           cortex (3, 42) and cerebral white matter closer to the cortex than to the lateral
                  ventricles (juxtacortical and subcortical); lobe from SynthSeg --parc if it exists,
                  else not reported
  other           hippocampus (17, 48), amygdala (18, 45), CSF (5, 6, 7? no: 24 + ventricles 4, 5, 6, 43, 44),
                  outside the brain (0). Reported, but NOT used for the CAA patterns.

Sensitivity analysis (pre-specified): a fixed 5 mm juxtacortical band (distance to cortex <= 5 mm) instead
of the white-matter distance rule.

`classify_lesion(seg, mask, voxels_mm, band_mm=5.0)` returns the class for one lesion mask.
The distance is computed from the LESION VOXELS to the nearest voxel of each region, so a 1-voxel
microbleed in the periventricular white matter lands deep and one under the cortex lands lobar.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)

# SynthSeg (FSL 6.x / FreeSurfer numbering, as used by cmb.stage2.candidates)
CEREBELLUM = (7, 8, 46, 47)
BRAINSTEM = (16,)
DEEP_GRAY = (10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60)
WM = (2, 41)
CORTEX = (3, 42)
LAT_VENT = (4, 43, 5, 44)          # lateral ventricles incl. inferior horns (5, 44); 3rd/4th (6, 7) are CSF too
CSF = (4, 5, 6, 7, 43, 44, 24)     # ventricles + free CSF
OTHER = (17, 18, 45, 48)           # hippocampus, amygdala
CLASSES = ("lobar", "deep", "infratentorial", "other")


def _distance_to(labels: np.ndarray, target: Sequence[int], mask: np.ndarray) -> float:
    """Minimum Euclidean distance (in voxels) from any voxel of `mask` to any voxel with a target label."""
    tgt = np.isin(labels, list(target))
    if not tgt.any():
        return float("inf")
    both = tgt & mask
    if both.any():
        return 0.0
    # distance from the mask to the region (and back is the same min); one map is enough
    d = ndimage.distance_transform_edt(~tgt)
    return float(d[mask].min()) if mask.any() else float("inf")


def classify_lesion(seg: np.ndarray, mask: np.ndarray, voxels_mm: Tuple[float, float, float],
                    band_mm: float = 5.0) -> Dict:
    """-> dict(mars, sub, dist_vent_mm, dist_cortex_mm, band5mm_lobar, labels_in_lesion).

    `seg` SynthSeg on the lesion grid, `mask` bool lesion mask on the same grid, `voxels_mm` (x,y,z) spacing.
    """
    mask = np.asarray(mask, bool)
    vox = np.argwhere(mask)
    if len(vox) == 0:
        raise ValueError("empty lesion mask")

    labels_in = np.unique(seg[mask]).tolist()
    # majority label over the lesion voxels (ties: smaller label wins, deterministic).
    # Fixed 11.11.2026: counted the UNIQUE labels before (each label once), so the smallest label present won.
    counts = np.bincount(np.asarray(seg[mask]).ravel())
    majority = int(np.argmax(counts))

    if majority in CEREBELLUM:
        return dict(mars="infratentorial", sub="cerebellum", dist_vent_mm=float("nan"),
                    dist_cortex_mm=float("nan"), band5mm_lobar=False, labels_in_lesion=labels_in)
    if majority in BRAINSTEM:
        return dict(mars="infratentorial", sub="brainstem", dist_vent_mm=float("nan"),
                    dist_cortex_mm=float("nan"), band5mm_lobar=False, labels_in_lesion=labels_in)
    if majority in DEEP_GRAY:
        return dict(mars="deep", sub=None, dist_vent_mm=float("nan"),
                    dist_cortex_mm=float("nan"), band5mm_lobar=False, labels_in_lesion=labels_in)
    if majority in OTHER:
        return dict(mars="other", sub=None, dist_vent_mm=float("nan"),
                    dist_cortex_mm=float("nan"), band5mm_lobar=False, labels_in_lesion=labels_in)
    if majority in CSF or majority == 0:
        return dict(mars="other", sub="outside_csf", dist_vent_mm=float("nan"),
                    dist_cortex_mm=float("nan"), band5mm_lobar=False, labels_in_lesion=labels_in)
    if majority in CORTEX:
        return dict(mars="lobar", sub=None, dist_vent_mm=float("nan"),
                    dist_cortex_mm=0.0, band5mm_lobar=True, labels_in_lesion=labels_in)
    if majority in WM:
        d_vent = _distance_to(seg, LAT_VENT, mask)
        d_cort = _distance_to(seg, CORTEX, mask)
        d_vent_mm = d_vent * min(voxels_mm[0], voxels_mm[1]) if np.isfinite(d_vent) else float("nan")
        d_cort_mm = d_cort * min(voxels_mm[0], voxels_mm[1]) if np.isfinite(d_cort) else float("nan")
        mars = "deep" if (np.isfinite(d_vent) and d_vent < d_cort) else "lobar"
        band5 = bool(np.isfinite(d_cort_mm) and d_cort_mm <= band_mm)
        return dict(mars=mars, sub=None, dist_vent_mm=d_vent_mm, dist_cortex_mm=d_cort_mm,
                    band5mm_lobar=band5, labels_in_lesion=labels_in)
    # any other label: treat as other (e.g. 3rd/4th ventricle not in CSF set, or 15/19 etc.)
    return dict(mars="other", sub=f"label_{majority}", dist_vent_mm=float("nan"),
                dist_cortex_mm=float("nan"), band5mm_lobar=False, labels_in_lesion=labels_in)


def lesions_of(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    labels, n = ndimage.label(mask, structure=CONNECTIVITY_26)
    return labels, int(n)


if __name__ == "__main__":
    # synthetic self-test: 20x20x20 block, one WM voxel ring, one cortex voxel ring, one lesion each side
    seg = np.zeros((20, 20, 20), np.int32)
    seg[:, :, 10] = 43                       # lateral ventricle slab
    seg[:, :, 0] = 3                         # cortex slab
    seg[10:14, :, 11:15] = 2                 # WM block next to the ventricle
    seg[10:14, :, 1:4] = 2                   # WM block next to the cortex
    deep = np.zeros((20, 20, 20), bool); deep[11, 10, 12] = True
    lobar = np.zeros((20, 20, 20), bool); lobar[11, 10, 2] = True
    cereb = np.zeros((20, 20, 20), np.int32); cereb[:] = 7
    r1 = classify_lesion(seg, deep, (1, 1, 1)); r2 = classify_lesion(seg, lobar, (1, 1, 1))
    r3 = classify_lesion(cereb, lobar, (1, 1, 1))
    assert r1["mars"] == "deep", r1
    assert r2["mars"] == "lobar" and r2["band5mm_lobar"], r2
    assert r3["mars"] == "infratentorial" and r3["sub"] == "cerebellum", r3
    print("OK deep", r1)
    print("OK lobar", r2)
    print("OK cerebellum", r3)
