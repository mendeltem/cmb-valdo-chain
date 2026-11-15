"""wk2b (2026-11-05): the P->FT arm of cmb.stage2.pooled (--pooled-finetune, label 'pooled-ft <model>').

CPU unit test (run with plain `python cmb/tests/test_pooled_finetune.py` -- the
dl env has no pytest; exit 1 on failure). It swaps the trackset MODELS for a recording proxy (same
contract as fit_predict_trackset: positional train/test bundles + keyword pretrain_x/pretrain_y/
finetune_epochs/finetune_lr/pretrain_cache) and checks the two leakage invariants of ARBEIT.md ## wk2b:

  1. the pre-training set never contains a case of fold k of ANY cohort (inner passes: also none of
     fold j while fold j is held out);
  2. the fine-tuning set is ONLY the target cohort D's cases with fold != k (inner passes: D's
     cases with fold != k and != j).

The proxy records the bundle case_index values (pooled.py's sorted case ids 0..N-1); the expected
sets are rebuilt at CASE level from candidates.json (case -> fold, cohorts in the order pooled.py
concatenates them), so the checks hold for both views and copies. The proxy scores are deterministic
functions of (seed, case), so the full pooled.main() run -- mode registration, 5-fold loop,
threshold selection, per-case JSON -- executes for real on CPU without a torch training run.
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.tests.test_tset_apply import make_set          # noqa: E402

N_CALLS = []                                           # dict(seed, train, test, pretrain, finetune_epochs, finetune_lr)


def _proxy():
    """Recording stand-in for fit_predict_trackset with the wk2b keyword contract included."""
    def fit(train_x, train_y, test_x, seed=0, pretrain_x=None, pretrain_y=None,
            finetune_epochs=None, finetune_lr=None, pretrain_cache=None, **kw):
        N_CALLS.append(dict(
            seed=int(seed),
            train=np.unique(np.asarray(train_x.case_index, np.int64)),
            test=np.unique(np.asarray(test_x.case_index, np.int64)),
            pretrain=None if pretrain_x is None else np.unique(np.asarray(pretrain_x.case_index, np.int64)),
            n_pretrain=0 if pretrain_y is None else int(np.sum(np.asarray(pretrain_y) == 1)),
            finetune_epochs=finetune_epochs, finetune_lr=finetune_lr))
        return np.array([((int(seed) * 131 + i + 7) % 97) / 97.0 for i in np.asarray(test_x.case_index, np.int64)], np.float32)
    return fit


def _run(cohorts, tmp):
    from cmb.stage2 import pooled, trackset
    saved = trackset.MODELS
    trackset.MODELS = {"tset-plain": _proxy()}
    saved_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        sys.argv = ["pooled", "--cohort", *[f"{n}={tmp}/{n}" for n in cohorts],
                    "--models", "tset-plain", "--skip-own", "--pooled-finetune",
                    "--ft-epochs", "2", "1", "--seed-offset", "0", "--json", f"{tmp}/out.json"]
        pooled.main()
    finally:
        trackset.MODELS = saved
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_env
    return json.load(open(f"{tmp}/out.json"))


def _case_layout(cohorts, tmp):
    """case_index (pooled.py's sorted case id) -> (cohort, fold), cohorts in the order pooled.py
    concatenates them (the order of the --cohort args), cases sorted per cohort."""
    layout = []
    for n in cohorts:
        info = json.load(open(f"{tmp}/{n}/candidates.json"))
        for cid in sorted(info["cases"]):
            layout.append((n, info["cases"][cid]["fold"]))
    return layout


def test_pooled_finetune_invariants():
    """For each target D and fold k: pre set has NO fold-k case of any cohort; fine set = D's cases fold != k."""
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as tmp:
        cohorts = ["a", "b"]
        for n in cohorts:
            make_set(f"{tmp}/{n}", n, 10, rng)
        N_CALLS.clear()
        d = _run(cohorts, tmp)
        label = "pooled-ft tset-plain"
        assert label in d["results"], f"missing mode {label}; got {sorted(d['results'])}"
        for coh in cohorts:
            m = d["results"][label][coh]
            assert "f1" in m and np.isfinite(m["f1"]) and 0.0 <= m["f1"] <= 1.0, (label, coh, m)
        layout = _case_layout(cohorts, tmp)
        # 2 targets x 5 folds x (4 inner + 1 final) = 50 calls with a pretrain set
        ft = [c for c in N_CALLS if c["pretrain"] is not None]
        assert len(ft) == 50, f"expected 50 P->FT calls, got {len(ft)}"
        for c in ft:
            assert c["finetune_epochs"] == 1 and c["finetune_lr"] == 5e-4, c   # half of 2 ep / lr 1e-3
        for t, target in enumerate(cohorts):
            for k in range(5):
                # final fit: seed = k + 1000*t + 1000*seed_offset; test = D's fold-k cases
                # (the bundle is built only from D's own candidates, so test = D, fold == k)
                test_expect = np.array(sorted(i for i, (co, fl) in enumerate(layout) if co == target and fl == k), np.int64)
                finals = [c for c in ft if c["seed"] == k + 1000 * t and np.array_equal(c["test"], test_expect)]
                assert len(finals) == 1, (f"final call target {target} fold {k}: "
                                          f"{[(c['seed'], list(c['test'][:4])) for c in ft if c['seed'] == k + 1000 * t]}")
                sel = finals[0]
                bad = [i for i in sel["pretrain"] if layout[i][1] == k]
                assert not bad, f"pre-train set target {target} fold {k} contains fold-k cases: {[layout[i] for i in bad[:5]]}"
                # pooled set: every case of EVERY cohort with fold != k, no others
                expect_pre = np.array(sorted(i for i, (co, fl) in enumerate(layout) if fl != k), np.int64)
                assert np.array_equal(sel["pretrain"], expect_pre), f"pre set target {target} fold {k} != all cohorts fold != k"
                # fine-tuning set = ONLY D's own cases with fold != k
                expect_fine = np.array(sorted(i for i, (co, fl) in enumerate(layout) if co == target and fl != k), np.int64)
                assert np.array_equal(sel["train"], expect_fine), f"fine set target {target} fold {k} != D fold != k"


def test_pooled_finetune_inner_passes():
    """Inner passes (threshold selection) must be equally honest: pre set holds out fold k AND fold j,
    fine set = D's cases with fold != k and fold != j."""
    rng = np.random.RandomState(1)
    with tempfile.TemporaryDirectory() as tmp:
        cohorts = ["a"]
        make_set(f"{tmp}/a", "a", 10, rng)
        N_CALLS.clear()
        _run(cohorts, tmp)
        layout = _case_layout(cohorts, tmp)
        for k in range(5):
            for j in range(5):
                if j == k:
                    continue
                # inner fit: test = ALL fold-j cases (of every cohort, in the single-cohort test layout
                # this is D's fold-j cases); seed = k*100 + j + 50
                inner_expect = np.array(sorted(i for i, (_, fl) in enumerate(layout) if fl == j), np.int64)
                sel = [c for c in N_CALLS if c["pretrain"] is not None and c["seed"] == k * 100 + j + 50
                       and np.array_equal(c["test"], inner_expect)]
                assert len(sel) == 1, f"inner call k={k} j={j}: {[ (c['seed'], list(c['test'][:3])) for c in N_CALLS if c['seed'] == k * 100 + j + 50]}"
                sel = sel[0]
                bad = [i for i in sel["pretrain"] if layout[i][1] in (k, j)]
                assert not bad, f"inner pre-train set k={k} j={j} contains held-out cases: {[layout[i] for i in bad[:5]]}"
                expect_fine = np.array(sorted(i for i, (_, fl) in enumerate(layout) if fl != k and fl != j), np.int64)
                assert np.array_equal(sel["train"], expect_fine), f"inner fine set k={k} j={j} != D fold != k,j"


if __name__ == "__main__":
    test_pooled_finetune_invariants()
    print("ok test_pooled_finetune_invariants")
    test_pooled_finetune_inner_passes()
    print("ok test_pooled_finetune_inner_passes\nALL 2 TESTS OK")
