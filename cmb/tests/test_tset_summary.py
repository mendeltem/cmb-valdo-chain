"""cmb.stage2.tset_summary (rules v2) on synthetic pooled JSONs: two real arms carry, eight null arms do not; screening; JSON shape.

    python -m cmb.tests.test_tset_summary          (run from the repository root)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.stage2 import tset_summary as S      # noqa: E402

COHORTS = {"valdo": 57, "t2star": 75, "swi": 61}


def write_runs(tmp, n_seeds, effects, seed=1):
    """effects: arm -> probability per case that one false positive disappears (0 = null arm)."""
    rng = np.random.RandomState(seed)
    base = {c: {f"{c}{i}": (rng.poisson(2.5), rng.poisson(1.5), rng.poisson(1.2)) for i in range(n)} for c, n in COHORTS.items()}
    files = []
    for s in range(n_seeds):
        per_case = {}
        for arm, gain in effects.items():
            per_case[arm] = {c: {k: [int(v[0]), max(int(v[1] - (rng.rand() < gain) + (rng.rand() < 0.15) - (rng.rand() < 0.15)), 0), int(v[2])]
                                 for k, v in d.items()} for c, d in base.items()}
        f = os.path.join(tmp, f"s{s}.json"); json.dump(dict(seed_offset=s, per_case=per_case), open(f, "w")); files.append(f)
    return files


def test_real_arms_carry_null_arms_do_not():
    effects = {"pooled tset-plain": 0.0, "pooled real-a": 0.7, "pooled real-b": 0.8}
    effects.update({f"pooled null-{j}": 0.0 for j in range(8)})
    with tempfile.TemporaryDirectory() as tmp:
        res = S.summarise(S.load(write_runs(tmp, 2, effects)), "pooled tset-plain", draws=800)
    carried = sorted(m for m, e in res.items() if e["decision"] == "carry to confirmation")
    assert carried == ["pooled real-a", "pooled real-b"], {m: (e["delta_all"], e["simultaneous_lower"]) for m, e in res.items()}
    e = res["pooled real-a"]
    assert e["sigma_seed"] >= S.SIGMA_SEED_FLOOR and e["se_total"] > e["se_case"] and 0 < e["shrunk_delta"] < e["delta_all"]
    assert set(e["delta_cohorts"]) == set(COHORTS) and set(e["ci95_cohorts"]) == set(COHORTS)


def test_screening_with_one_seed():
    with tempfile.TemporaryDirectory() as tmp:
        res = S.summarise(S.load(write_runs(tmp, 1, {"pooled tset-plain": 0.0, "pooled good": 0.8, "pooled bad": -0.0})), "pooled tset-plain", draws=200)
    assert res["pooled good"]["decision"] == "advance to seed 1"
    assert res["pooled bad"]["decision"] == "stop (screening)"


def test_tag_separates_runs_on_another_candidate_set():
    with tempfile.TemporaryDirectory() as tmp:
        base = write_runs(tmp, 1, {"pooled tset-plain": 0.0}, seed=1)
        os.makedirs(f"{tmp}/u")
        union = write_runs(f"{tmp}/u", 1, {"pooled tset-plain": 0.8}, seed=1)
        runs = S.load(base + union, {union[0]: "union"})
        assert set(m for _, m in runs) == {"pooled tset-plain", "pooled tset-plain union"}
        res = S.summarise(runs, "pooled tset-plain", draws=200)
        assert res["pooled tset-plain union"]["decision"] == "advance to seed 1"


def test_harm_in_one_cohort_blocks_carry():
    e = dict(seeds=[0, 1], delta_all=0.03, delta_cohorts={"a": 0.05, "b": -0.01}, ci95_cohorts={"a": [0.02, 0.08], "b": [-0.04, -0.001]},
             se_case=0.008, simultaneous_lower=0.01)
    assert S.decide(e) == "drop"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__, flush=True)
    print(f"ALL {len(tests)} TESTS OK")
