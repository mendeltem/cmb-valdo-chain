"""Agreement of per-patient CMB counts between the reference and the model: ICC(3,1).

ICC(3,1) of Shrout & Fleiss (1979), = ICC(C,1) of McGraw & Wong (1996): two-way mixed model, consistency, single
rater. From the two-way ANOVA of the n x k table (n patients, k raters):

    MSR = k * sum_i (mean_i - grand)^2 / (n - 1)          between patients
    MSE = (SS_total - SS_patients - SS_raters) / ((n - 1)(k - 1))
    ICC(3,1) = (MSR - MSE) / (MSR + (k - 1) MSE)

The paper scripts (scripts/paper/phase1_arena/auswertung.py, phase2_finetune/good_for_us.py,
phase2_finetune/wk4b_update_summary.py, phase3_cohorts/wk5_klinisch.py) had their own copies until 16.11.2026. Those
multiplied the between-patient sum of squares by n instead of k, did not divide it by n - 1 and used n - 1 instead of
k - 1 in the denominator, which gives about 0.997 for any data; they now import this function (CHANGES.md).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def icc_3_1(table: np.ndarray | Sequence[Sequence[float]]) -> float:
    """ICC(3,1) of an n x k table (rows = patients, columns = raters). NaN for fewer than 3 patients, fewer than
    2 raters, or a table without any variation (the coefficient is undefined there, not 0 or 1)."""
    y = np.asarray(table, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"ICC(3,1) needs an n x k table (patients x raters), got shape {y.shape}")
    n, k = y.shape
    if n < 3 or k < 2:
        return float("nan")
    grand = y.mean()
    ss_rows = k * float(np.sum((y.mean(axis=1) - grand) ** 2))
    ss_cols = n * float(np.sum((y.mean(axis=0) - grand) ** 2))
    ss_err = float(np.sum((y - grand) ** 2)) - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = max(ss_err, 0.0) / ((n - 1) * (k - 1))  # rounding can push an exact 0 slightly below it
    denom = ms_rows + (k - 1) * ms_err
    if denom <= 0:
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def icc31(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """ICC(3,1) of two raters, e.g. reference and model CMB count per patient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"two equally long 1-D count vectors needed, got {x.shape} and {y.shape}")
    return icc_3_1(np.stack([x, y], axis=1))
