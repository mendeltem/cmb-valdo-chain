"""Schleife R02 (2026-11-04): the three new CPU stage-2 precision arms of cmb.stage2.pooled
(feat-vessel-atlas, feat-ens-nb, feat-vessel-tissue) register, run on CPU without a GPU,
and produce finite, deterministic scores on a tiny synthetic candidate set -- both with and
without venous-atlas fields (the atlas arm must degrade to zeros, not crash, on candidate sets
that were not built with cmb.stage2.atlas_features). Also a regression guard that the R01 arm
feat-vessel is still registered and runs unchanged when the R02 arms are registered alongside.

    python -m cmb.tests.test_schleife_r02          (run from the repository root)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.tests.test_tset_apply import make_set          # noqa: E402

R02_ARMS = ("feat-vessel-atlas", "feat-ens-nb", "feat-vessel-tissue")


def _run_pooled(folder, models, out):
    from cmb.stage2 import pooled
    saved = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""               # force CPU
    sys.argv = ["pooled", "--cohort", f"a={folder}", "--models", *models, "--tissue",
                "--skip-own", "--seed-offset", "0", "--json", out]
    try:
        pooled.main()
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return json.load(open(out))


def _make(folder, seed, with_atlas):
    make_set(folder, "a", 12, np.random.RandomState(seed))
    if with_atlas:
        info = json.load(open(f"{folder}/candidates.json"))
        rng = np.random.RandomState(seed + 1)
        for r in info["candidates"]:
            r["atlas_vein_p"] = float(rng.rand())
            r["atlas_vein_max3mm"] = float(rng.rand())
            r["atlas_vein_dist_mm"] = float(rng.rand() * 20.0)
        json.dump(info, open(f"{folder}/candidates.json", "w"))


def test_r02_arms_run_and_are_finite():
    for with_atlas in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            folder = f"{tmp}/a"; _make(folder, 0, with_atlas)
            d = _run_pooled(folder, list(R02_ARMS), f"{tmp}/r02.json")
            for arm in R02_ARMS:
                key = f"pooled {arm}"
                assert key in d["results"], f"missing mode {key}; got {sorted(d['results'])}"
                for coh, m in d["results"][key].items():
                    if isinstance(m, dict) and "f1" in m:
                        assert np.isfinite(m["f1"]) and 0.0 <= m["f1"] <= 1.0


def test_r02_arms_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/b"; _make(folder, 1, True)
        d1 = _run_pooled(folder, list(R02_ARMS), f"{tmp}/d1.json")
        d2 = _run_pooled(folder, list(R02_ARMS), f"{tmp}/d2.json")
        for arm in R02_ARMS:
            a = d1["results"][f"pooled {arm}"]["a"]; b = d2["results"][f"pooled {arm}"]["a"]
            assert abs(a["f1"] - b["f1"]) < 1e-6, (arm, a, b)


def test_atlas_arm_differs_from_vessel_when_atlas_present():
    """With real atlas values the feature MATRIX of the atlas arm must differ from feat-vessel's
    (otherwise the features silently do nothing); without them it must equal it. Checking the
    F1 would be meaningless on 12 synthetic cases, so we capture the training matrix shape via
    a fit proxy installed in a copy of FEATURE_MODELS."""
    from cmb.stage2 import pooled
    for with_atlas, expect_same in ((True, False), (False, True)):
        with tempfile.TemporaryDirectory() as tmp:
            folder = f"{tmp}/c"; _make(folder, 2, with_atlas)
            captured = []
            proxy = lambda train_x, train_y, test_x, seed=0: \
                (captured.append(np.asarray(train_x).shape), np.zeros(len(test_x), np.float32))[1]
            orig = pooled.FEATURE_MODELS
            pooled.FEATURE_MODELS = {
                "feat-vessel": ("neighbour", proxy),
                "feat-vessel-atlas": ("neighbour-atlas", proxy),
            }
            try:
                _run_pooled(folder, ["feat-vessel", "feat-vessel-atlas"], f"{tmp}/v.json")
            finally:
                pooled.FEATURE_MODELS = orig
            assert len(captured) == 10, f"expected 2 arms x 5 folds, got {len(captured)}"
            vessel_shape, atlas_shape = captured[0][1], captured[5][1]
            same = vessel_shape == atlas_shape
            assert same == expect_same, (with_atlas, vessel_shape, atlas_shape)


def test_r01_arm_registration_guard():
    """The R01 arm feat-vessel still runs when the R02 arms are registered alongside."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/d"; _make(folder, 3, False)
        d = _run_pooled(folder, ["feat-vessel"] + list(R02_ARMS), f"{tmp}/g.json")
        assert "pooled feat-vessel" in d["results"]
        for arm in R02_ARMS:
            assert f"pooled {arm}" in d["results"]
