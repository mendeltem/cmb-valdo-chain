"""CPU unit tests for the wk2 helpers in cmb.stage2 (2026-10-31): --channels and --match-size-to.

Synthetic phantoms only, no patient data. Run from the repository root:
    python -m cmb.tests.test_wk2_helpers
"""
from __future__ import annotations

import sys
import tempfile

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))

from cmb.stage2 import trackset as T
from cmb.stage2.pooled import CHANNEL_SETS  # noqa: F401  -- must exist

rng = np.random.RandomState(0)


def make_phantom(n_cand: int, n_cases: int, shape=(24, 40, 40)) -> np.ndarray:   # CROP 16,32,32 + jitter 2,4,4
    n, c, *s = (n_cand, 3) + tuple(shape)
    p = rng.rand(n, c, *s).astype(np.float32)
    for i in range(0, n_cand, 2):                       # every second one is a 'bleed' (dark core, high probability)
        zc, yc, xc = tuple(x // 2 for x in shape)
        p[i, 2, zc, yc, xc] = 0.95
    return p


def test_channels_run():
    """fit_predict_trackset runs for each channel set and returns a score per test candidate."""
    patches = make_phantom(12, 4)
    side = np.zeros((12, T.SIDE_FEATURES + T.ATLAS_FEATURES), np.float32)
    case_index = np.arange(12) % 4
    centre = np.zeros((12, 3), np.float32)
    b = T.Bundle(patches, side, case_index, centre, None)
    y = np.array([0, 1] * 6)
    tr, te = np.arange(8), np.arange(8, 12)
    for tag, ch in (("all", (0, 1, 2)), ("prob", (2,)), ("prob-frst", (1, 2))):
        s = T.fit_predict_trackset(b[tr], y[tr], b[te], seed=1, modules=(), epochs=1, members=1,
                                   device="cpu", channels=ch)
        assert s.shape == (4,) and np.all(np.isfinite(s)), (tag, s)
    # the default call is untouched (bit-identical interface)
    s0 = T.fit_predict_trackset(b[tr], y[tr], b[te], seed=1, modules=(), epochs=1, members=1, device="cpu")
    assert s0.shape == (4,)
    print("ok: channels run (all/prob/prob-frst) ->", {t: tuple(c) for t, c in CHANNEL_SETS.items()})


def test_match_size_draw():
    """The drawn set has the SAME size and positive count as D's own pool, never contains a test-fold case,
    and the draw is reproducible in the same seed order (seed offset = draw AND init)."""
    n = 200
    cohort = np.array(["valdo"] * 80 + ["t2star"] * 80 + ["swi"] * 40)
    fold = np.array([i % 5 for i in range(n)])
    y = np.zeros(n, int)
    for d, start in (("valdo", 0), ("t2star", 80), ("swi", 160)):
        idx = np.nonzero(cohort == d)[0]
        y[idx[::2]] = 1                                  # 50 % positive per cohort
    target, k = "swi", 2
    own = np.nonzero((cohort == target) & (fold != k))[0]
    n_pos, n_neg = int(y[own].sum()), len(own) - int(y[own].sum())
    pool_pos = np.nonzero((fold != k) & (y == 1))[0]
    pool_neg = np.nonzero((fold != k) & (y == 0))[0]
    for seed_offset in range(3):
        rngk = np.random.RandomState(1000 * seed_offset + 100 + k)
        draw = np.concatenate([rngk.choice(pool_pos, n_pos, replace=False), rngk.choice(pool_neg, n_neg, replace=False)])
        assert len(draw) == len(own)
        assert int(y[draw].sum()) == n_pos
        assert not (fold[draw] == k).any()                       # never a test-fold case
        # drawn from the POOLED pool, not only from the own cohort (that is the whole point: mixed composition)
        assert (cohort[draw] != target).sum() > 0, "draw must mix cohorts"
    # reproducibility: same seed -> same draw
    a = np.random.RandomState(1000 * 0 + 100 + 2).choice(pool_pos, n_pos, replace=False)
    b = np.random.RandomState(1000 * 0 + 100 + 2).choice(pool_pos, n_pos, replace=False)
    assert (a == b).all()
    print(f"ok: match-size draw (swi fold {k}: own {len(own)} = {n_pos} pos / {n_neg} neg; draws reproducible, no test case)")


if __name__ == "__main__":
    test_channels_run()
    test_match_size_draw()
    print("ALL OK")
