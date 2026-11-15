"""cmb.maskfix on a synthetic mask with one enclosed hole and one open notch.

The enclosed hole (a 1x1x2 cavity inside the brain, closed in 3D and in every axial
slice) is filled; the open notch (a 2-voxel-wide channel cut from the image border all
the way through the brain, open to the border) is NOT a hole and stays 0. Also: the
hole definition must match holes_of in qc/build_model_input_qc.py (3D fill plus axial
2D fill), and fill_case must write <mask>_filled.nii.gz with the original header/dtype
and leave a hole-free mask untouched.

    python -m pytest -q cmb/tests/test_maskfix.py
"""
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cmb.maskfix import fill as maskfix  # noqa: E402

AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])


def make_mask():
    """32x32x24 volume. Brain = box x 8..23, y 8..23, z 4..19.
    Hole: 2 voxels at (14,14,10..11) -- enclosed in 3D and in every axial slice.
    Notch: channel x 0..23, y 18..19, z 10..10 -- open to the image border at both ends."""
    m = np.zeros((32, 32, 24), dtype=np.uint8)
    m[8:24, 8:24, 4:20] = 1
    m[14, 14, 10:12] = 0          # enclosed hole (1x1x2 = 2 voxels)
    m[0:24, 18:20, 10] = 0        # open notch through the brain
    return m


def test_hole_filled_notch_kept():
    m = make_mask()
    out = maskfix.fill_mask(m)
    # enclosed hole is filled
    assert int(out[14, 14, 10:12].sum()) == 2
    # open notch is NOT filled
    assert int(out[0:24, 18:20, 10].sum()) == 0
    # nothing else changed
    m2 = m.copy(); m2[14, 14, 10:12] = 1
    assert (out.astype(bool) == m2.astype(bool)).all()


def test_holes_of_matches_qc_definition():
    """The hole set must equal the definition in qc/build_model_input_qc.py:
    3D fill plus 2D fill of every axial slice, minus the brain."""
    m = make_mask().astype(bool)
    expected = ndimage.binary_fill_holes(m)
    expected |= np.stack([ndimage.binary_fill_holes(m[..., k]) for k in range(m.shape[2])], -1)
    expected &= ~m
    assert (maskfix.holes_of(m) == expected).all()
    assert int(expected.sum()) == 2          # exactly the enclosed hole, not the notch
    # a cavity with a tunnel open to the border along z is NOT a 3D hole, but IS
    # enclosed in every axial slice -> the 2D rule fills it (as in the QC)
    m2 = np.zeros((16, 16, 16), bool); m2[3:13, 3:13, 3:13] = 1
    m2[5:11, 5:11, 4:12] = 0                 # cavity
    m2[8, 8, 0:4] = 0                        # 1-voxel tunnel through the surface to the border
    h = maskfix.holes_of(m2)
    assert h[8, 8, 8].astype(bool)
    assert not ndimage.binary_fill_holes(m2)[8, 8, 8]       # 3D alone would miss it


def test_fill_case_writes_filled_nifti():
    m = make_mask()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "sub-001_desc-brain_mask.nii.gz")
        nib.save(nib.Nifti1Image(m, AFFINE), p)
        out = maskfix.fill_case(p)
        assert out == p[:-len(".nii.gz")] + "_filled.nii.gz"
        d = np.asarray(nib.load(out).dataobj)
        assert d.dtype == m.dtype
        assert (d[14, 14, 10:12] > 0).all()
        assert int(d[0:24, 18:20, 10].sum()) == 0
        assert np.allclose(nib.load(out).affine, AFFINE)
        # idempotent: second call returns the existing file
        assert maskfix.fill_case(p) == out


def test_fill_case_holefree_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        m = np.ones((10, 10, 10), dtype=np.uint8)
        p = os.path.join(tmp, "x_mask.nii.gz")
        nib.save(nib.Nifti1Image(m, AFFINE), p)
        out = maskfix.fill_case(p)
        d = np.asarray(nib.load(out).dataobj)
        assert (d == 1).all()
        assert maskfix.fill_case(p) == out


if __name__ == "__main__":
    test_hole_filled_notch_kept()
    test_holes_of_matches_qc_definition()
    test_fill_case_writes_filled_nifti()
    test_fill_case_holefree_untouched()
    print("ok all 4 tests")
    print("ALL 4 TESTS OK")
