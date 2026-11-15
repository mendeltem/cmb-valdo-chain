"""Schleife R03 (2026-11-04): the three new CPU stage-2 precision arms of cmb.stage2.pooled
(feat-vessel-tissue-atlas, feat-ens-nb-tissue, feat-mine-nb-tissue) register, run on CPU
without a GPU, and produce finite, deterministic scores on a tiny synthetic candidate set --
both with and without venous-atlas fields (the tissue-atlas arm must degrade to zeros, not
crash, on candidate sets that were not built with cmb.stage2.atlas_features). Also a
regression guard that the R02 best-candidate arm feat-vessel-tissue is still registered and
runs unchanged when the R03 arms are registered alongside.

    python -m cmb.tests.test_schleife_r03          (run from the repository root)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.tests.test_tset_apply import make_set          # noqa: E402

R03_ARMS = ("feat-vessel-tissue-atlas", "feat-ens-nb-tissue", "feat-mine-nb-tissue")


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


def test_r03_arms_run_and_are_finite():
    for with_atlas in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            folder = f"{tmp}/a"; _make(folder, 0, with_atlas)
            d = _run_pooled(folder, list(R03_ARMS), f"{tmp}/r03.json")
            for arm in R03_ARMS:
                key = f"pooled {arm}"
                assert key in d["results"], f"missing mode {key}; got {sorted(d['results'])}"
                for coh, m in d["results"][key].items():
                    if isinstance(m, dict) and "f1" in m:
                        assert np.isfinite(m["f1"]) and 0.0 <= m["f1"] <= 1.0


def test_r03_arms_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/b"; _make(folder, 1, True)
        d1 = _run_pooled(folder, list(R03_ARMS), f"{tmp}/d1.json")
        d2 = _run_pooled(folder, list(R03_ARMS), f"{tmp}/d2.json")
        for arm in R03_ARMS:
            a = d1["results"][f"pooled {arm}"]["a"]; b = d2["results"][f"pooled {arm}"]["a"]
            assert abs(a["f1"] - b["f1"]) < 1e-6, (arm, a, b)


def test_tissue_atlas_arm_matrix_differs_from_tissue_when_atlas_present():
    """With real atlas values the 3 atlas COLUMNS of the tissue-atlas arm must be non-zero
    (the features actually reach the model); without them they must be all zeros (the arm
    degrades to feat-vessel-tissue, never crashes). We capture the first training matrix
    through a fit proxy installed in a copy of FEATURE_MODELS; each arm is run ALONE."""
    from cmb.stage2 import pooled
    for with_atlas in (False, True):
        mats = {}
        for arm in ("feat-vessel-tissue", "feat-vessel-tissue-atlas"):
            with tempfile.TemporaryDirectory() as tmp:
                folder = f"{tmp}/c"; _make(folder, 2, with_atlas)
                captured = []
                proxy = lambda train_x, train_y, test_x, seed=0: \
                    (captured.append(np.asarray(train_x, dtype=np.float64).copy()),
                     np.zeros(len(test_x), np.float32))[1]
                orig = pooled.FEATURE_MODELS
                orig_kind, _ = (pooled.FEATURE_MODELS_R03[arm] if arm in pooled.FEATURE_MODELS_R03
                                else pooled.FEATURE_MODELS_R02[arm])
                pooled.FEATURE_MODELS = {arm: (orig_kind, proxy)}
                try:
                    _run_pooled(folder, [arm], f"{tmp}/v.json")
                finally:
                    pooled.FEATURE_MODELS = orig
                assert captured, f"no fit calls captured for {arm}"
                mats[arm] = captured[0]
        base, atlas = mats["feat-vessel-tissue"], mats["feat-vessel-tissue-atlas"]
        n = min(base.shape[0], atlas.shape[0])
        assert atlas.shape[1] == base.shape[1] + 3, (base.shape, atlas.shape)
        assert np.allclose(base[:n], atlas[:n, :base.shape[1]]), "base part must be identical"
        extra = atlas[:n, base.shape[1]:]
        if with_atlas:
            assert not np.allclose(extra, 0.0), "atlas columns are all zeros although the set has atlas values"
        else:
            assert np.allclose(extra, 0.0), "atlas columns must be zeros on a set without atlas values"


def test_r02_arm_registration_guard():
    """The R02 best-candidate arm feat-vessel-tissue still runs when the R03 arms are
    registered alongside (old arms unchanged)."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/d"; _make(folder, 3, False)
        d = _run_pooled(folder, ["feat-vessel-tissue"] + list(R03_ARMS), f"{tmp}/g.json")
        assert "pooled feat-vessel-tissue" in d["results"]
        for arm in R03_ARMS:
            assert f"pooled {arm}" in d["results"]


if __name__ == "__main__":
    test_r03_arms_run_and_are_finite()
    test_r03_arms_deterministic()
    test_tissue_atlas_arm_matrix_differs_from_tissue_when_atlas_present()
    test_r02_arm_registration_guard()
    print("all R03 arm tests passed")
