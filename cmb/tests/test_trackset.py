"""Tests for cmb.stage2.trackset and its hook in cmb.stage2.pooled -- synthetic phantoms only, CPU, no patient data.

    python -m cmb.tests.test_trackset          (run from the repository root, ~2 min on CPU)

Phantoms: the image channel is inverted like the grids (microbleeds and vessels bright). A 'microbleed' is a ball of radius
2 voxels, a 'vessel' a tube of radius 1.5 voxels that crosses every slice and drifts in-plane by one voxel per slice.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.stage2 import trackset as T          # noqa: E402

M = T._modules()


def phantom(kind: str, rng: np.random.RandomState, shape=(20, 40, 40), drift=(0, 1)) -> np.ndarray:
    """(3, z, y, x) float16 patch: image, a crude 'FRST' (the image smoothed), 'probability' (image > 0.5 near the centre)."""
    Z, Y, X = shape
    z, y, x = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
    zc, yc, xc = Z // 2, Y // 2, X // 2
    if kind == "bleed":
        inside = ((z - zc) / 2.0) ** 2 + ((y - yc) / 4.0) ** 2 + ((x - xc) / 4.0) ** 2 <= 1.0
    else:
        cy = yc + drift[0] * (z - zc); cx = xc + drift[1] * (z - zc)
        inside = (y - cy) ** 2 + (x - cx) ** 2 <= 3.0 ** 2
    img = 0.3 + 0.05 * rng.randn(Z, Y, X).astype(np.float32) + 0.6 * inside
    from scipy import ndimage
    frst = ndimage.gaussian_filter(img, 1.5)
    prob = np.where(((z - zc) ** 2 <= 4) & ((y - yc) ** 2 + (x - xc) ** 2 <= 16), img, 0.0)
    return np.stack([img, frst, prob]).astype(np.float16)


def test_d4_group_axioms():
    table, inverse = T.d4_tables()
    assert all(table[0][g] == g == table[g][0] for g in range(8))
    assert all(table[table[a][b]][c] == table[a][table[b][c]] for a in range(8) for b in range(8) for c in range(8))
    assert all(table[g][inverse[g]] == 0 for g in range(8))
    assert sorted(set(table[1][g] for g in range(8))) == list(range(8))


def test_group_conv_is_equivariant():
    torch.manual_seed(0)
    table, inverse = T.d4_tables()
    lift = M["D4Conv3d"](3, 4, lift=True).double(); conv = M["D4Conv3d"](4, 5).double()
    x = torch.randn(1, 3, 6, 12, 12, dtype=torch.float64)
    f = conv(lift(x))
    for a in range(8):
        fa = conv(lift(T.d4_apply(x, a))).view(1, 5, 8, 6, 12, 12)
        ref = T.d4_apply(f, a).view(1, 5, 8, 6, 12, 12)
        expected = torch.stack([ref[:, :, table[inverse[a]][g]] for g in range(8)], 2)
        assert torch.allclose(fa, expected, atol=1e-9), a


def test_d4_encoder_and_net_are_invariant():
    torch.manual_seed(0)
    net = M["TrackSetNet"](("d4",)).double().eval()
    x = torch.randn(2, 3, 16, 32, 32, dtype=torch.float64)
    side = torch.zeros(2, T.SIDE_FEATURES + T.ATLAS_FEATURES, dtype=torch.float64); case = torch.tensor([0, 0]); centre = torch.zeros(2, 3, dtype=torch.float64)
    with torch.no_grad():
        ref, _ = net(x, side, case, centre)
        for g in range(8):
            out, _ = net(T.d4_apply(x, g), side, case, centre)
            assert torch.allclose(out, ref, atol=1e-8), (g, out, ref)
    plain = M["TrackSetNet"](()).double().eval()
    with torch.no_grad():                           # the control is NOT invariant -- otherwise the test would prove nothing
        assert not torch.allclose(plain(x, side, case, centre)[0], plain(T.d4_apply(x, 1), side, case, centre)[0], atol=1e-6)


def test_tracking_follows_a_vessel_and_loses_a_bleed():
    rng = np.random.RandomState(0)
    tube, ball = phantom("vessel", rng), phantom("bleed", rng)
    tr = T.track_candidates(np.stack([tube, ball]))
    f = tr["feats"]
    half = T.TRACK_HALF
    assert tr["crops"].shape == (2, 2 * half + 1, 3, T.TRACK_CROP, T.TRACK_CROP)
    # the tube drifts one voxel (0.5 mm) in x per slice: the track must follow it
    assert np.allclose(f[0, half + 4, 1], 4 * 0.5, atol=0.5) and np.allclose(f[0, half - 4, 1], -4 * 0.5, atol=0.5)
    # contrast stays high along the vessel and drops beyond the ball
    assert f[0, :, 2][f[0, :, 3] > 0].min() > 0.3
    assert f[1, half, 2] > 0.3 and max(f[1, half + 5, 2], f[1, half - 5, 2]) < 0.15
    # 20 slices cover all 2 * 9 + 1 tracked ones; a 12-slice patch leaves the outer slices marked missing
    assert f[0, :, 3].all()
    short = T.track_candidates(phantom("vessel", rng, shape=(12, 40, 40))[None])["feats"][0]
    assert short[0, 3] == 0 and short[-1, 3] == 0 and short[half, 3] == 1 and short[half + 5, 3] == 1


def test_tta8_prediction_is_invariant():
    """With 'tta8' the test score is the mean over the 8 in-plane views, so rotating every test input changes nothing."""
    rng = np.random.RandomState(4)
    train, ytr = _synthetic_bundle(4, rng); test, _ = _synthetic_bundle(2, rng)
    rot = T.Bundle(np.stack([T.d4_apply(torch.from_numpy(p.astype(np.float32)), 1).numpy() for p in test.patches]).astype(np.float16),
                   test.side, test.case_index, test.centre_mm, None)
    rot.tracks = T.track_candidates(rot.patches)
    a = T.fit_predict_trackset(train, ytr, test, seed=3, modules=("track", "tta8"), epochs=1, members=1, device="cpu")
    b = T.fit_predict_trackset(train, ytr, rot, seed=3, modules=("track", "tta8"), epochs=1, members=1, device="cpu")
    assert np.allclose(a, b, atol=1e-5), (a, b)


def test_offsets_rotate_with_the_patch():
    rng = np.random.RandomState(1)
    tube = phantom("vessel", rng, drift=(1, 0))
    for g in range(8):
        rotated = T.d4_apply(torch.from_numpy(tube.astype(np.float32)), g).numpy().astype(np.float16)
        a = T.track_candidates(tube[None])["feats"][0, :, :2]
        b = T.track_candidates(rotated[None])["feats"][0, :, :2]
        assert np.allclose(a @ T.d4_offset_matrix(g).T, b, atol=1e-5), g


def test_neighbour_features():
    centre = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [60, 0, 0], [0, 0, 0]], np.float32)
    case = np.array([0, 0, 0, 0, 1]); p = np.array([0.2, 0.9, 0.4, 0.5, 0.3], np.float32)
    nb = T.neighbour_features(case, centre, p)
    assert nb[1, 1] == 2 and abs(nb[1, 2] - 0.4) < 1e-6 and abs(nb[1, 3] - 1.0) < 1e-6      # middle bead: between two
    assert nb[0, 3] == 0.0                                                                      # end bead: neighbours on one side
    assert nb[3, 1] == 0 and nb[4, 0] == 0.0                                                    # far away / alone in its scan


def test_side_features_from_rows():
    rng = np.random.RandomState(6)
    p = np.stack([phantom("bleed", rng), phantom("bleed", rng)])
    p[1, 0, :, :, :14] = 0                                   # brain ends 6 voxels = 3 mm left of the centre (x = 20)
    rows = [dict(probability=0.4, voxels=20, tissue=2), dict(probability=0.9, voxels=60, tissue=24, n_runs=3, atlas_vein_p=0.7,
                                                                   atlas_vein_max3mm=0.9, atlas_vein_dist_mm=1.5)]
    s = T.side_from_rows(rows, p, {2: 2, 24: 0})
    assert s.shape == (2, T.SIDE_FEATURES + T.ATLAS_FEATURES)
    assert s[0, 2] == 1 and s[1, 0] == 1 and s[1, 7] == 0.5 and s[0, 7] == 0
    assert s[0, 8] == 1.0 and abs(s[1, 8] * T.EDGE_CAP_MM - 3.5) < 0.6, s[1, 8] * T.EDGE_CAP_MM
    assert np.allclose(s[1, 9:], [0.7, 0.9, 0.075]) and (s[0, 9:11] == 0).all()


def test_bundle_slicing():
    b = T.Bundle(np.arange(5 * 2).reshape(5, 2), np.ones((5, 7)), [0, 0, 1, 1, 2], np.zeros((5, 3)),
                 dict(crops=np.arange(5), feats=np.arange(5) * 2))
    s = b[np.array([4, 1])]
    assert len(s) == 2 and s.case_index.tolist() == [2, 0] and s.tracks["feats"].tolist() == [8, 2]
    assert len(b[np.array([True, False, True, False, False])]) == 2 and b.shape == (5, 2)


def _synthetic_bundle(n_cases: int, rng, with_tracks=True):
    patches, y, case, centre = [], [], [], []
    for c in range(n_cases):
        for j in range(4):
            bleed = (c + j) % 2 == 0
            patches.append(phantom("bleed" if bleed else "vessel", rng, drift=(rng.randint(-1, 2), rng.randint(-1, 2))))
            y.append(int(bleed)); case.append(c); centre.append([j * 12.0, 0, 0])
    patches = np.stack(patches)
    side = np.zeros((len(y), T.SIDE_FEATURES + T.ATLAS_FEATURES), np.float32)
    return T.Bundle(patches, side, case, centre, T.track_candidates(patches) if with_tracks else None), np.array(y)


def _auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def test_models_learn_bleeds_versus_vessels():
    rng = np.random.RandomState(2)
    train, ytr = _synthetic_bundle(10, rng); test, yte = _synthetic_bundle(4, rng)
    for modules in [(), ("d4", "proto"), ("track", "ctx", "set"), ("wide", "tta8"), ("track", "tta8"), ("ctx", "atlas")]:
        s = T.fit_predict_trackset(train, ytr, test, seed=1, modules=modules, epochs=6, members=1, device="cpu")
        assert s.shape == (len(test),) and np.isfinite(s).all()
        assert _auc(s, yte) > 0.8, (modules, _auc(s, yte))


def test_pooled_runs_trackset_arms_end_to_end():
    """cmb.stage2.pooled with a tiny synthetic cohort: tset arms and --neighbours run and land in the JSON."""
    import functools
    from cmb.stage2 import pooled
    rng = np.random.RandomState(3)
    with tempfile.TemporaryDirectory() as tmp:
        cohort = os.path.join(tmp, "cohort"); os.makedirs(cohort)
        patches, rows, cases = [], [], {}
        for c in range(15):
            cid = f"case{c:02d}"; n_ref = 0
            for j in range(3):
                bleed = (c + j) % 3 == 0
                patches.append(phantom("bleed" if bleed else "vessel", rng))
                rows.append(dict(case=cid, fold=c % 5, touched=[j + 1] if bleed else [], voxels=40 + j, probability=0.5,
                                 tissue=2, centre_voxel=[20 * j, 10, 5]))
                n_ref += bleed
            cases[cid] = dict(fold=c % 5, n_reference=n_ref)
        np.save(os.path.join(cohort, "patches.npy"), np.stack(patches))
        json.dump(dict(threshold=0.15, candidates=rows, cases=cases), open(os.path.join(cohort, "candidates.json"), "w"))
        saved = dict(T.MODELS)
        try:
            for name, mods in T.ARMS.items():
                T.MODELS[name] = functools.partial(T.fit_predict_trackset, modules=mods, epochs=1, members=1, device="cpu")
            out = os.path.join(tmp, "out.json")
            sys.argv = ["pooled", "--cohort", f"syn={cohort}", "--models", "logistic", "tset-plain", "tset-full", "tset-atlas",
                        "--skip-own", "--neighbours", "--json", out]
            pooled.main()
        finally:
            T.MODELS.clear(); T.MODELS.update(saved)
        res = json.load(open(out))["results"]
        for label in ("pooled tset-plain", "pooled tset-full", "pooled tset-atlas", "pooled+nb logistic", "pooled logistic"):
            assert label in res and "f1" in res[label]["syn"], label


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__, flush=True)
    print(f"ALL {len(tests)} TESTS OK")
