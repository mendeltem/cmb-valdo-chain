"""Image loading and connected components for the command-line layer, with the types stated.

``nibabel.load`` is typed as returning any file-based image and ``scipy.ndimage.label`` as returning an int or a
tuple, so a type checker cannot see that the command line uses voxel images and (labels, count) pairs. These two
helpers say it once and check it at run time; values are exactly those of the wrapped calls.
"""

from __future__ import annotations

import os

import numpy as np
from nibabel.spatialimages import SpatialImage


def load(path: str | os.PathLike) -> SpatialImage:
    """``nibabel.load`` restricted to voxel images (NIfTI, MGH, ...); anything else stops the run."""
    import nibabel as nib

    img = nib.load(path)
    if not isinstance(img, SpatialImage):
        raise SystemExit(f"{path}: not a voxel image ({type(img).__name__})")
    return img


def components(mask: np.ndarray, structure: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """``scipy.ndimage.label``: (label volume, number of components)."""
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=structure)  # type: ignore[misc]  # (array, int) when no output is given
    return labels, int(n)
