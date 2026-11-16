"""ICC(3,1) of cmb.agreement, used by the paper scripts for the per-patient CMB count agreement.

The paper version's own copies returned about 0.997 for any data (between-patient sum of squares times n instead of
k, not divided by n - 1, n - 1 instead of k - 1 in the denominator; CHANGES.md). These tests would have caught it:
a textbook value, an independent ANOVA, and data with poor agreement that must not come out near 1.
"""

import numpy as np
import pytest

from cmb.agreement import icc31, icc_3_1

# Shrout & Fleiss (1979), Psychol Bull 86:420, Table 2: 6 targets x 4 judges. Their Table 4 gives ICC(3,1) = .71
# (and ICC(1,1) = .17, ICC(2,1) = .29 for the other models).
SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def wrong_copy(x, y):
    """The formula of the paper version's own icc31 copies, kept to show what the tests catch."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    g = np.stack([x, y], axis=1)
    ss_b = n * np.sum((g.mean(axis=1) - g.mean()) ** 2)
    ss_w = np.sum((g - g.mean(axis=1, keepdims=True)) ** 2)
    msb, msw = ss_b / 1, ss_w / (n - 1)
    return float((msb - msw) / (msb + (n - 1) * msw))


def test_shrout_fleiss_table_2():
    assert icc_3_1(SHROUT_FLEISS) == pytest.approx(0.7148, abs=5e-4)


def test_two_rater_form_matches_the_table_form():
    rng = np.random.default_rng(0)
    x = rng.poisson(4, 40).astype(float)
    y = x + rng.normal(0, 2, 40)
    assert icc31(x, y) == pytest.approx(icc_3_1(np.stack([x, y], 1)))


def test_poor_agreement_is_not_near_one():
    rng = np.random.default_rng(1)
    x = rng.poisson(3, 60).astype(float)
    y = rng.poisson(3, 60).astype(float)  # independent of x: true ICC 0
    assert abs(icc31(x, y)) < 0.35
    assert wrong_copy(x, y) > 0.95  # the old copy: near 1 anyway


def test_opposite_order_is_negative():
    x = np.arange(10, dtype=float)
    assert icc31(x, x[::-1]) == pytest.approx(-1.0)


def test_consistency_ignores_a_constant_offset():
    x = np.array([0, 1, 3, 5, 12, 2, 7], float)
    assert icc31(x, x + 4) == pytest.approx(1.0)
    assert icc31(x, x) == pytest.approx(1.0)


def test_undefined_cases_are_nan():
    assert np.isnan(icc31([1, 2], [1, 2]))  # fewer than 3 patients
    assert np.isnan(icc31([2, 2, 2, 2], [2, 2, 2, 2]))  # no variation at all
    with pytest.raises(ValueError):
        icc31([1, 2, 3], [1, 2])


def test_against_a_two_way_anova_of_statsmodels():
    sm = pytest.importorskip("statsmodels.formula.api")
    anova_lm = pytest.importorskip("statsmodels.stats.anova").anova_lm
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    for table in (SHROUT_FLEISS, np.stack([t := rng.poisson(5, 30).astype(float), t + rng.normal(0, 3, 30)], 1)):
        n, k = table.shape
        df = pd.DataFrame(
            {
                "y": table.ravel(),
                "subject": np.repeat(np.arange(n), k).astype(str),
                "rater": np.tile(np.arange(k), n).astype(str),
            }
        )
        aov = anova_lm(sm.ols("y ~ C(subject) + C(rater)", df).fit())
        msr = aov.loc["C(subject)", "mean_sq"]
        mse = aov.loc["Residual", "mean_sq"]
        assert icc_3_1(table) == pytest.approx((msr - mse) / (msr + (k - 1) * mse), rel=1e-9)


def test_against_pingouin():
    pg = pytest.importorskip("pingouin")
    pd = pytest.importorskip("pandas")
    n, k = SHROUT_FLEISS.shape
    df = pd.DataFrame({"y": SHROUT_FLEISS.ravel(), "t": np.repeat(np.arange(n), k), "r": np.tile(np.arange(k), n)})
    res = pg.intraclass_corr(df, targets="t", raters="r", ratings="y").set_index("Type")["ICC"]
    label = "ICC3" if "ICC3" in res.index else "ICC(C,1)"  # pingouin < 0.7 / >= 0.7 names of the same row
    assert icc_3_1(SHROUT_FLEISS) == pytest.approx(res[label], rel=1e-9)
