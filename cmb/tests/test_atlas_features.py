"""cmb.stage2.atlas_features on synthetic data (CPU, no FreeSurfer run, no patient data).

    python -m cmb.tests.test_atlas_features          (run from the repository root)
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.stage2 import atlas_features as A      # noqa: E402


def test_feature_values():
    vein = np.zeros((40, 40, 20), np.float32)
    vein[20, :, 10] = 0.9                                  # a vein along y at x = 20, z = 10
    v = A.features_for_case(vein, np.array([[20, 5, 10], [24, 5, 10], [35, 5, 10]]), level=0.5)
    assert np.isclose(v[0, 0], 0.9) and v[0, 2] == 0.0
    assert v[1, 0] == 0 and np.isclose(v[1, 1], 0.9) and np.isclose(v[1, 2], 2.0)      # 4 voxels x 0.5 mm = 2 mm away
    assert v[2, 1] == 0 and np.isclose(v[2, 2], 7.5)


def test_atlas_goes_to_the_template_grid_by_world_coordinates():
    """A blob at world (10, -20, 5) mm in a 0.6 mm atlas lands at the same world point on a 1 mm, x-flipped template grid."""
    with tempfile.TemporaryDirectory() as tmp:
        aff_a = np.diag([0.6, 0.6, 0.6, 1.0]); aff_a[:3, 3] = [-30, -60, -30]
        a = np.zeros((100, 100, 100), np.float32); ijk = np.round((np.array([10, -20, 5]) - aff_a[:3, 3]) / 0.6).astype(int)
        a[tuple(slice(i - 3, i + 4) for i in ijk)] = 1.0
        nib.save(nib.Nifti1Image(a, aff_a), f"{tmp}/atlas.nii.gz")
        aff_t = np.diag([-1.0, 1.0, 1.0, 1.0]); aff_t[:3, 3] = [40, -70, -40]
        nib.save(nib.Nifti1Image(np.zeros((80, 90, 80), np.float32), aff_t), f"{tmp}/tpl.nii.gz")
        out = nib.load(A.atlas_on_template(f"{tmp}/atlas.nii.gz", f"{tmp}/tpl.nii.gz", f"{tmp}/o.nii.gz"))
        d = np.asarray(out.dataobj)
        centre = np.argwhere(d > 0.5).mean(0)                     # centre of the plateau, not the first maximum
        world = out.affine @ np.r_[centre, 1]
        assert np.allclose(world[:3], [10, -20, 5], atol=0.75), world


def run(argv):
    sys.argv = ["atlas"] + argv
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.main()
    return buf.getvalue()


def test_features_folder_and_auc():
    with tempfile.TemporaryDirectory() as tmp:
        cand = f"{tmp}/cand"; os.makedirs(cand); maps = f"{tmp}/maps"; os.makedirs(maps)
        rows, cases = [], {}
        for c in range(3):
            cid = f"sub-{c}"
            vein = np.zeros((40, 40, 20), np.float32); vein[20, :, 10] = 0.9
            nib.save(nib.Nifti1Image(vein, np.diag([0.5, 0.5, 1, 1])), f"{maps}/{cid}_vein.nii.gz")
            rows += [dict(case=cid, fold=c, touched=[], voxels=10, probability=0.4, centre_voxel=[20, 7, 10]),
                     dict(case=cid, fold=c, touched=[1], voxels=10, probability=0.4, centre_voxel=[33, 7, 10])]
            cases[cid] = dict(fold=c, n_reference=1)
        json.dump(dict(threshold=0.15, candidates=rows, cases=cases), open(f"{cand}/candidates.json", "w"))
        np.save(f"{cand}/patches.npy", np.zeros((6, 3, 4, 4, 4), np.float16))
        run(["features", "--candidates", cand, "--maps", f"{maps}/{{id}}_vein.nii.gz", "--out", f"{tmp}/out"])
        d = json.load(open(f"{tmp}/out/candidates.json"))
        assert all(set(A.FIELDS) <= set(r) for r in d["candidates"]) and os.path.islink(f"{tmp}/out/patches.npy")
        assert d["candidates"][0]["atlas_vein_p"] > 0.8 and d["candidates"][1]["atlas_vein_p"] == 0
        text = run(["auc", "--candidates", f"syn={tmp}/out"])
        assert "atlas_vein_p 1.000" in text, text                 # every false candidate sits on the vein here
        dry = run(["register", "--grid", f"{tmp}/grid", "--cases-from", cand, "--out", f"{tmp}/reg", "--dry-run"])
        assert dry.count("mri_synthmorph register") == 3 and dry.count("mri_synthmorph apply") == 3 and A.TEMPLATE in dry


if __name__ == "__main__":
    for t in (test_feature_values, test_atlas_goes_to_the_template_grid_by_world_coordinates, test_features_folder_and_auc):
        t(); print("ok", t.__name__, flush=True)
    print("ALL 3 TESTS OK")
