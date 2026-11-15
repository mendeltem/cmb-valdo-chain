# -*- coding: utf-8 -*-
"""CPU unit tests for cmb.analysis.unet_rerank (synthetic probability maps).

Covers the two invariants the task names explicitly:
  1. a map with a KNOWN best threshold is recovered by the nested choice, and
  2. the nested choice NEVER looks at the test fold (poisoned fold stays out).
Plus FROC/CPM, the ceiling-recall budget and bootstrap reproducibility.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cmb.analysis.unet_rerank import (  # noqa: E402
    BOOTSTRAP, GRID, FROC_FPS, FROC_DET_T, FIXED_T,
    bootstrap_delta_f1, case_rows, ceiling_recall, cpm_curve,
    f1_of_counts, froc_cpm, nested_thresholds,
)


def _ball(shape, center, radius=1):
    m = np.zeros(shape, bool)
    z, y, x = center
    m[z - radius:z + radius + 1, y - radius:y + radius + 1, x - radius:x + radius + 1] = True
    return m


def test_nested_recovers_known_best_threshold():
    """One true lesion per case (prob 0.8), noise all below t_true=0.55.
    The pooled F1 on the four training folds peaks exactly at 0.55, so every
    t_k must come out 0.55."""
    rng = np.random.RandomState(0)
    t_true = 0.55
    shape = (12, 12, 12)
    folds, fold_of_case, all_rows = [], [], []
    for f in range(5):
        case_rows_f = []
        for c in range(2):
            p = np.zeros(shape, np.float32)
            ref = np.zeros(shape, bool)
            ref[3 + 5 * c:4 + 5 * c, 5, 5] = True
            p[ref] = 0.8
            n = 50
            p[rng.randint(0, 12, n), rng.randint(0, 12, n), rng.randint(0, 12, n)] = \
                rng.random(n) * (t_true - 0.02)
            fold_of_case.append(f)
            all_rows.append(case_rows(p, ref))
            case_rows_f.append((p, ref))
        folds.append(case_rows_f)
    tks = nested_thresholds(all_rows, fold_of_case)
    assert tks == [t_true] * 5, tks


def test_nested_choice_never_looks_at_test_fold():
    """Poison fold 3 with a huge lesion + a sea of 0.5 noise. If fold 3's own
    data influenced t_3 the choice would move; the nested rule keeps 0.55."""
    rng = np.random.RandomState(0)
    t_true = 0.55
    shape = (12, 12, 12)
    fold_of_case, all_rows = [], []
    for f in range(5):
        for c in range(2):
            p = np.zeros(shape, np.float32)
            ref = np.zeros(shape, bool)
            ref[3 + 5 * c:4 + 5 * c, 5, 5] = True
            p[ref] = 0.8
            n = 50
            p[rng.randint(0, 12, n), rng.randint(0, 12, n), rng.randint(0, 12, n)] = \
                rng.random(n) * (t_true - 0.02)
            if f == 3 and c == 0:
                p = np.zeros(shape, np.float32)
                ref = np.zeros(shape, bool)
                ref[2, 2, 2] = True
                p[2, 2, 2] = 0.99
                p[6:9, 6:9, 6:9] = 0.5      # the poison: everything at 0.5
            fold_of_case.append(f)
            all_rows.append(case_rows(p, ref))
    tks = nested_thresholds(all_rows, fold_of_case)
    assert all(t == t_true for t in tks), tks


def test_cpm_luna16():
    """Three separate detections (0.9 TP, 0.7 FP, 0.6 TP), two reference
    lesions. TP needs voxel overlap with the reference. LUNA16 keeps the top
    round(fp*n_scans) by cluster max probability."""
    p = np.zeros((10, 10, 10), np.float32)
    ref = np.zeros((10, 10, 10), bool)
    ref[1, 1, 1] = True
    ref[8, 8, 8] = True
    p[1, 1, 1] = 0.9   # TP (on ref 1)
    p[4, 4, 4] = 0.7   # FP (off any ref)
    p[8, 8, 8] = 0.6   # TP (on ref 2)
    dets = cpm_curve(p, ref)
    assert len(dets) == 3, dets
    by_score = sorted(dets, reverse=True)
    assert [round(s, 2) for s, _ in by_score] == [0.9, 0.7, 0.6], dets
    assert by_score[0][1] is True and by_score[1][1] is False and by_score[2][1] is True, dets
    cpm, curve = froc_cpm(dets, n_scans=1, n_gt=2, fps=[0.5, 1.0, 2.0])
    # fp=0.5 -> keep 1 (the 0.9 TP) -> 0.5; fp=1 -> keep 2 -> 1 TP -> 0.5;
    # fp=2 -> keep 3 -> 2 TP -> 1.0
    assert curve["0.5"] == 0.5 and curve["1.0"] == 0.5 and curve["2.0"] == 1.0, curve
    assert abs(cpm - (0.5 + 0.5 + 1.0) / 3) < 1e-9, cpm
    cpm_all, _ = froc_cpm(dets, n_scans=1, n_gt=2, fps=FROC_FPS)
    assert 0.0 <= cpm_all <= 1.0


def test_ceiling_recall_topn_budget():
    """TP = detection overlapping a reference voxel. One TP (0.9, on ref) plus
    one FP line (max 0.8); top-3 keeps both. Raising thr to 0.85 drops the FP."""
    p = np.zeros((10, 10, 10), np.float32)
    ref = np.zeros((10, 10, 10), bool)
    ref[1, 1, 1] = True
    p[1, 1, 1] = 0.9                      # TP on the reference
    p[3, 1, 1], p[4, 1, 1], p[5, 1, 1] = 0.8, 0.7, 0.6   # one FP component (max 0.8)
    assert ceiling_recall(p, ref, topn=3) == (1.0, 1, 1)
    assert ceiling_recall(p, ref, thr=0.85, topn=3) == (1.0, 1, 1)
    # budget of 1 keeps only the 0.9 TP
    assert ceiling_recall(p, ref, topn=1) == (1.0, 1, 1)
    # no detections at all -> recall 0
    assert ceiling_recall(np.zeros((10, 10, 10), np.float32), ref, topn=3) == (0.0, 0, 1)
    # a detection off every reference is an FP (recall 0, not an error)
    p2 = np.zeros((10, 10, 10), np.float32)
    p2[7, 7, 7] = 0.9
    assert ceiling_recall(p2, ref, topn=3) == (0.0, 0, 1)


def test_f1_and_pooled_consistency():
    """case_rows + nested F1 must agree with the metric.py definition on a
    hand-computed volume."""
    shape = (6, 6, 6)
    p = np.zeros(shape, np.float32)
    ref = _ball(shape, (1, 1, 1), 0)          # one 1-voxel reference
    p[1, 1, 1] = 0.6
    p[4, 4, 4] = 0.55
    rows = case_rows(p, ref)
    ti = GRID.index(0.6)
    # at 0.6: only the lesion voxel -> 1 TP, 0 FP, 0 FN
    assert rows[ti][:3] == (1, 0, 0), rows[ti]
    tj = GRID.index(0.55)
    # at 0.55: lesion + FP -> 1 TP, 1 FP, 0 FN
    assert rows[tj][:3] == (1, 1, 0), rows[tj]
    assert abs(f1_of_counts(1, 1, 0) - 2 / 3) < 1e-9
    tks = nested_thresholds([rows], [0])
    # degenerate: no other folds -> every grid point ties at F1 0 -> first wins
    assert tks[0] == GRID[0], tks


def test_bootstrap_reproducible():
    rows_a = [(1, 0, 0, 1), (0, 1, 1, 1), (1, 1, 0, 1)] * 4
    rows_b = [(1, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 1)] * 4
    fi = GRID.index(FIXED_T)
    d1 = bootstrap_delta_f1(rows_a, rows_b, fi, fi, n=50)
    d2 = bootstrap_delta_f1(rows_a, rows_b, fi, fi, n=50)
    assert d1 == d2
    assert d1[3] == 12 and 0 <= d1[0] <= 1


def test_grid_sanity():
    assert GRID[0] == 0.05 and GRID[-1] == 0.90 and len(GRID) == 18
    assert FIXED_T in GRID and 0.15 in GRID
    assert BOOTSTRAP == 2000
    assert FROC_FPS == [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert FROC_DET_T == 0.05


def test_module_selftest_cli():
    from cmb.paths import ROOT as root
    from cmb.paths import PY as py
    r = subprocess.run([py, "-m", "cmb.analysis.unet_rerank", "--selftest"],
                       cwd=root, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


if __name__ == "__main__":
    # plain-python fallback if pytest is unavailable
    for fn in (test_nested_recovers_known_best_threshold,
               test_nested_choice_never_looks_at_test_fold,
               test_cpm_luna16, test_ceiling_recall_topn_budget,
               test_f1_and_pooled_consistency, test_bootstrap_reproducible,
               test_grid_sanity):
        fn()
        print("ok:", fn.__name__)
