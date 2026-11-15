"""Schleife R04 (2026-11-04): the three new CPU stage-2 precision arms of cmb.stage2.pooled
(feat-ens-tissue-atlas, feat-mine-tissue-atlas, and the pooled+extra synthetic-mimic arm)
register, run on CPU without a GPU, and produce finite, deterministic scores on a tiny
synthetic candidate set. The tissue-atlas arms must degrade to zeros (not crash) on
candidate sets without venous-atlas fields. A regression guard checks that the R03
best-candidate arm feat-vessel-tissue-atlas still runs unchanged when the R04 arms are
registered alongside.

    python -m cmb.tests.test_schleife_r04          (run from the repository root)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.tests.test_tset_apply import make_set          # noqa: E402

R04_ARMS = ("feat-ens-tissue-atlas", "feat-mine-tissue-atlas")
R03_BEST = "feat-vessel-tissue-atlas"


def _run_pooled(folder, models, out, extra=None):
    from cmb.stage2 import pooled
    saved = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""               # force CPU
    argv = ["pooled", "--cohort", f"a={folder}", "--models", *models, "--tissue",
            "--skip-own", "--seed-offset", "0", "--json", out]
    if extra is not None:
        argv += ["--extra", f"a={extra}"]
    sys.argv = argv
    try:
        pooled.main()
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return json.load(open(out))


def _make(folder, seed, with_atlas, prefix="a"):
    make_set(folder, prefix, 12, np.random.RandomState(seed))
    if with_atlas:
        info = json.load(open(f"{folder}/candidates.json"))
        rng = np.random.RandomState(seed + 1)
        for r in info["candidates"]:
            r["atlas_vein_p"] = float(rng.rand())
            r["atlas_vein_max3mm"] = float(rng.rand())
            r["atlas_vein_dist_mm"] = float(rng.rand() * 20.0)
        json.dump(info, open(f"{folder}/candidates.json", "w"))


def test_r04_arms_run_and_are_finite():
    """Both R04 arms run on CPU, on the tissue-atlas basis, and give a finite F1 in [0,1]."""
    for with_atlas in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            folder = f"{tmp}/a"; _make(folder, 0, with_atlas)
            d = _run_pooled(folder, list(R04_ARMS), f"{tmp}/r04.json")
            for arm in R04_ARMS:
                key = f"pooled {arm}"
                assert key in d["results"], f"missing mode {key}; got {sorted(d['results'])}"
                m = d["results"][key]["a"]
                assert "f1" in m and np.isfinite(m["f1"]) and 0.0 <= m["f1"] <= 1.0, (arm, m)


def test_r04_arms_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/b"; _make(folder, 1, True)
        d1 = _run_pooled(folder, list(R04_ARMS), f"{tmp}/d1.json")
        d2 = _run_pooled(folder, list(R04_ARMS), f"{tmp}/d2.json")
        for arm in R04_ARMS:
            a = d1["results"][f"pooled {arm}"]["a"]; b = d2["results"][f"pooled {arm}"]["a"]
            assert abs(a["f1"] - b["f1"]) < 1e-6, (arm, a, b)


def test_tissue_atlas_arm_matrix_differs_from_tissue_when_atlas_present():
    """With real atlas values the 3 atlas COLUMNS of the R04 arms must be non-zero (the
    features actually reach the model); without them they must be all zeros (degrade to
    the tissue basis, never crash). We capture the first training matrix through a fit
    proxy installed in a copy of FEATURE_MODELS; each arm is run ALONE."""
    from cmb.stage2 import pooled
    for with_atlas in (False, True):
        mats = {}
        for arm in ("feat-vessel-tissue-atlas", "feat-ens-tissue-atlas", "feat-mine-tissue-atlas"):
            with tempfile.TemporaryDirectory() as tmp:
                folder = f"{tmp}/c"; _make(folder, 2, with_atlas)
                captured = []
                proxy = lambda train_x, train_y, test_x, seed=0: (
                    captured.append(np.asarray(train_x, dtype=np.float64).copy()),
                    np.zeros(len(test_x), np.float32))[1]
                # install a proxy fit for the arm so we can read its training matrix
                orig = dict(pooled.FEATURE_MODELS)
                if arm == "feat-mine-tissue-atlas":
                    # the mine arm's inputs are built in main(); we read them via the mode's
                    # `inputs` by swapping FEATURE_MODELS_R04's ensemble entry for the proxy
                    # is not possible for mine (fit=None). Instead capture through a wrapper
                    # by temporarily registering a plain feature model with the same kind.
                    pooled.FEATURE_MODELS[arm] = ("neighbour-tissue-atlas", proxy)
                else:
                    kind = pooled.FEATURE_MODELS[arm][0]
                    pooled.FEATURE_MODELS[arm] = (kind, proxy)
                try:
                    _run_pooled(folder, [arm], f"{tmp}/v.json")
                finally:
                    pooled.FEATURE_MODELS = orig
                assert captured, f"no fit calls captured for {arm}"
                mats[arm] = captured[0]
        base = mats["feat-vessel-tissue-atlas"]
        for arm in ("feat-ens-tissue-atlas", "feat-mine-tissue-atlas"):
            atlas = mats[arm]
            n = min(base.shape[0], atlas.shape[0])
            assert atlas.shape[1] == base.shape[1], (arm, base.shape, atlas.shape)
            assert np.allclose(base[:n], atlas[:n]), f"{arm} base part must be identical"
            if with_atlas:
                assert not np.allclose(base[:n, base.shape[1]-3:], 0.0), \
                    f"{arm}: atlas columns are all zeros although the set has atlas values"
            else:
                assert np.allclose(base[:n, base.shape[1]-3:], 0.0), \
                    f"{arm}: atlas columns must be zeros on a set without atlas values"


def test_r04_extra_arm_runs_and_is_finite():
    """pooled+extra <feat-vessel-tissue-atlas> with a synthetic mimic set (same cases and
    folds, training-only) runs on CPU and gives a finite F1; the extra candidates are never
    scored (the per-case count equals the main-set case count, not main+extra)."""
    with tempfile.TemporaryDirectory() as tmp:
        main, extra = f"{tmp}/a", f"{tmp}/ext"
        _make(main, 3, True); _make(extra, 4, True)      # same prefix 'a', same folds
        d = _run_pooled(main, [R03_BEST], f"{tmp}/syn.json", extra=extra)
        key = f"pooled+extra {R03_BEST}"
        assert key in d["results"], f"missing {key}; got {sorted(d['results'])}"
        m = d["results"][key]["a"]
        assert "f1" in m and np.isfinite(m["f1"]) and 0.0 <= m["f1"] <= 1.0, m
        n_cases = len({r["case"] for r in json.load(open(f"{main}/candidates.json"))["candidates"]})
        scored = sum(1 for c in d["per_case"][key]["a"])
        assert scored == n_cases, (scored, n_cases)     # extra candidates are not scored


def test_r03_best_arm_registration_guard():
    """The R03 best-candidate arm feat-vessel-tissue-atlas still runs when the R04 arms are
    registered alongside (old arms unchanged)."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/d"; _make(folder, 5, True)
        d = _run_pooled(folder, [R03_BEST] + list(R04_ARMS), f"{tmp}/g.json")
        assert f"pooled {R03_BEST}" in d["results"]
        for arm in R04_ARMS:
            assert f"pooled {arm}" in d["results"]


if __name__ == "__main__":
    test_r04_arms_run_and_are_finite()
    print("ok: arms run and finite")
    test_r04_arms_deterministic()
    print("ok: deterministic")
    test_tissue_atlas_arm_matrix_differs_from_tissue_when_atlas_present()
    print("ok: atlas columns reach the model")
    test_r04_extra_arm_runs_and_is_finite()
    print("ok: +extra (synthetic mimics) runs, extra not scored")
    test_r03_best_arm_registration_guard()
    print("ok: r03 best arm unchanged")
    print("ALL R04 TESTS PASSED")
