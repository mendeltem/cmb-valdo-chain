"""Schleife R01 (2026-11-04): the three new CPU feature-model arms of cmb.stage2.pooled
(feat-ens, feat-nb, feat-vessel) register, run on CPU without a GPU, and produce finite,
deterministic scores on a tiny synthetic candidate set. Also a regression guard that the
pre-existing 'pooled logistic' mode is still registered and runs.

    python -m cmb.tests.test_schleife_r01          (run from the repository root)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.tests.test_tset_apply import make_set          # noqa: E402


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


def _make(folder, seed):
    make_set(folder, "a", 12, np.random.RandomState(seed))


def test_r01_arms_run_and_are_finite():
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/a"; _make(folder, 0)
        d = _run_pooled(folder, ["feat-ens", "feat-nb", "feat-vessel"], f"{tmp}/r01.json")
        for arm in ("feat-ens", "feat-nb", "feat-vessel"):
            key = f"pooled {arm}"
            assert key in d["results"], f"missing mode {key}; got {sorted(d['results'])}"
            for coh, m in d["results"][key].items():
                if isinstance(m, dict) and "f1" in m:
                    assert np.isfinite(m["f1"]) and 0.0 <= m["f1"] <= 1.0


def test_r01_arms_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/b"; _make(folder, 1)
        d1 = _run_pooled(folder, ["feat-ens", "feat-nb", "feat-vessel"], f"{tmp}/d1.json")
        d2 = _run_pooled(folder, ["feat-ens", "feat-nb", "feat-vessel"], f"{tmp}/d2.json")
        for arm in ("feat-ens", "feat-nb", "feat-vessel"):
            a = d1["results"][f"pooled {arm}"]["a"]; b = d2["results"][f"pooled {arm}"]["a"]
            assert abs(a["f1"] - b["f1"]) < 1e-6, (arm, a, b)


def test_logistic_regression_guard():
    with tempfile.TemporaryDirectory() as tmp:
        folder = f"{tmp}/c"; _make(folder, 2)
        d = _run_pooled(folder, ["logistic"], f"{tmp}/lg.json")
        assert "pooled logistic" in d["results"]


if __name__ == "__main__":
    for t in (test_r01_arms_run_and_are_finite, test_r01_arms_deterministic, test_logistic_regression_guard):
        t()
        print(f"PASS {t.__name__}")
    print("all R01 arm tests passed")
