"""Physics checks of cmb.synthesis.dipole_render against known behaviour (CPU, no patient data).

    python -m cmb.tests.test_dipole_render          (run from the repository root)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.synthesis import dipole_render as R      # noqa: E402

GRID = (20, 40, 40); VOX = (1.0, 0.5, 0.5); MID = (10.0, 10.0, 10.0)


def bleed(chi=1.0, radius=1.0, w=0.0):
    return dict(kind="sphere", centre=MID, radius=radius, chi=chi, w_core=w)


def test_fields_match_the_formulas():
    z = np.array([0.0, 3.0, 0.0]); y = np.zeros(3); x = np.array([0.0, 0.0, 3.0])
    f, inside = R.field_sphere(z, y, x, (0, 0, 0), 1.0, 1.0, np.array([1.0, 0, 0]))
    assert inside[0] and f[0] == 0.0
    assert np.isclose(f[1], (1 / 3) * (1 / 27) * 2) and np.isclose(f[2], -(1 / 3) * (1 / 27))   # pole +2, equator -1
    fz, _ = R.field_cylinder(z, y, x, (0, 0, 0), (1, 0, 0), 0.5, 1.0, np.array([1.0, 0, 0]))   # parallel to B0
    assert fz[2] == 0.0 and np.isclose(fz[0], 1 / 3)
    fp, _ = R.field_cylinder(z, y, x, (0, 0, 0), (0, 1, 0), 0.5, 1.0, np.array([1.0, 0, 0]))   # perpendicular
    assert np.isclose(fp[0], -1 / 6) and np.isclose(fp[1], 0.5 * (0.25 / 9))


def test_attenuation_range_and_far_field():
    A = R.render(GRID, VOX, bleed(), 3.0, 20)
    assert A.dtype == np.float32 and A.min() >= 0 and A.max() <= 1
    assert A[10, 20, 20] < 0.2 and A[2, 2, 2] > 0.99


def test_magnitude_cannot_tell_calcium_from_blood():
    a = R.render(GRID, VOX, bleed(chi=1.2), 3.0, 20); b = R.render(GRID, VOX, bleed(chi=-1.2), 3.0, 20)
    assert np.allclose(a, b, atol=1e-5)


def test_blooming_grows_with_te_b0_and_slice_thickness():
    v = lambda **k: R.deficit_volume(R.render(GRID, VOX, bleed(), k.get("b0", 3.0), k.get("te", 20), acq_voxel_mm=k.get("acq")), VOX)
    v20, v40 = v(te=20), v(te=40)
    assert 1.3 < v40 / v20 < 2.5, v40 / v20                      # Liu 2012: x1.5-2.3 from TE 20 to 40 ms
    assert v(b0=3.0) > v(b0=1.5)
    assert v(acq=(3.0, 0.5, 0.5)) > 1.3 * v(acq=(1.0, 0.5, 0.5))


def test_vein_depends_on_its_angle_to_b0():
    """Parallel to B0 only the lumen offset acts; perpendicular the extravascular lobes dephase a wide zone."""
    def W(axis):
        o = dict(kind="cylinder", centre=MID, axis=axis, radius=0.4, chi=0.35, w_core=1.0)
        return R.deficit_volume(R.render(GRID, VOX, o, 3.0, 40, fine_mm=(0.1, 0.05, 0.05)), VOX)
    parallel, perpendicular = W((1, 0, 0)), W((0, 0, 1))
    assert perpendicular > 2 * parallel, (parallel, perpendicular)


def test_thick_slice_vein_is_shallower_but_bleed_grows():
    o = dict(kind="cylinder", centre=MID, axis=(0, 0, 1), radius=0.4, chi=0.35, w_core=1.0)
    thin = R.render(GRID, VOX, o, 3.0, 20, acq_voxel_mm=(1.0, 0.5, 0.5), fine_mm=(0.1, 0.05, 0.05))
    thick = R.render(GRID, VOX, o, 3.0, 20, acq_voxel_mm=(4.0, 0.5, 0.5), fine_mm=(0.1, 0.05, 0.05))
    assert thick.min() > thin.min()


def test_compose_keeps_background_and_brightens_the_object():
    rng = np.random.RandomState(0)
    host = np.zeros(GRID, np.float32); host[:, 5:35, 5:35] = 0.4 + 0.01 * rng.randn(20, 30, 30)
    A = R.render(GRID, VOX, bleed(radius=1.5), 3.0, 20)
    out = R.compose_inverted(host, A, rng)
    assert (out[host == 0] == 0).all()
    assert out[10, 20, 20] > host[10, 20, 20] + 0.3                 # inverted: the void becomes bright
    assert abs(out[2, 8, 8] - host[2, 8, 8]) < 0.05


def test_sampled_parameters_stay_in_range():
    rng = np.random.RandomState(1)
    for _ in range(200):
        b = R.sample_object("bleed", rng, MID)
        assert 0.5 <= b["radius"] <= 3.0 and 0.5 <= b["chi"] * 4 / 3 * np.pi * b["radius"] ** 3 <= 25
        v = R.sample_object("vein", rng, MID)
        assert 0.1 <= v["radius"] <= 0.75 and 0.15 < v["chi"] < 0.6 and np.isclose(np.linalg.norm(v["axis"]), 1)
        c = R.sample_object("calcification", rng, MID)
        assert c["chi"] < 0
    t = R.tilted_b0(rng); assert np.isclose(np.linalg.norm(t), 1) and t[0] >= np.cos(np.deg2rad(20)) - 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__, flush=True)
    print(f"ALL {len(tests)} TESTS OK")
