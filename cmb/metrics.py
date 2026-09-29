"""Lesion-level detection metrics for cerebral microbleeds -- both rules this project reports.

Replaces ``code/metric.py`` (touch rule) and the matching half of ``code/metric_valdo.py`` (official VALDO rule).
The old files stay in place while the GPU queue is calling them; this module is verified to reproduce them exactly
(``python -m cmb.tests.test_metrics_match_legacy``).

WHY TWO RULES, and why every number has to say which one it used
----------------------------------------------------------------
**Touch rule** (``hits``) -- the project's ranking measure. A reference lesion counts as found if ANY predicted
component overlaps it by at least one voxel; a prediction that touches no reference is a false positive. Several
predictions on one lesion count as one hit, not several. It is forgiving about outlines, which suits microbleeds of
3-8 voxels, where a one-voxel shift would otherwise decide the score.
Its weakness, stated openly: a single prediction touching TWO references counts as two hits and no false positive,
so one blob over the whole brain would score well. The size guard in the fixed post-processing (components above
2100 voxels are dropped) is what keeps that from happening, not the metric.

**VALDO rule** (``match_greedy``, ``case_valdo``) -- the official rule of the challenge, for comparison with
published numbers: 6-connectivity, threshold 0.5, and a ONE-TO-ONE match when the centres of mass are less than
5 mm apart. Stricter, and it punishes exactly the blob case above.

The two are not comparable. Section 11 of ARBEIT.md lists "F1" among the terms that must never be mixed: pooled
touch-rule F1 at the fixed operating point is the default in this project; per-subject VALDO F1 is reported
separately for the comparison with the challenge and with CenSynCMB.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)     # touch rule: diagonal neighbours count
CONNECTIVITY_6 = ndimage.generate_binary_structure(3, 1)   # VALDO rule: faces only
VALDO_THRESHOLD = 0.5
VALDO_DISTANCE_MM = 5.0


# --------------------------------------------------------------------------- touch rule
def hits(reference, prediction) -> Dict[str, int]:
    """Counts for one case under the touch rule.

    -> n_ref, n_pred, tp, fp, fn, case_ref, case_pred, count_error.
    ``tp`` counts REFERENCE components that were touched (not predictions), so several predictions on one lesion
    stay one hit; ``fp`` counts PREDICTED components that touch nothing.
    """
    reference = np.asarray(reference) > 0
    prediction = np.asarray(prediction) > 0
    ref_labels, n_ref = ndimage.label(reference, structure=CONNECTIVITY_26)
    pred_labels, n_pred = ndimage.label(prediction, structure=CONNECTIVITY_26)
    ref_hit = np.zeros(n_ref + 1, bool)
    pred_hit = np.zeros(n_pred + 1, bool)
    both = reference & prediction
    if both.any():
        ref_hit[np.unique(ref_labels[both])] = True
        pred_hit[np.unique(pred_labels[both])] = True
    tp = int(ref_hit[1:].sum())
    return dict(n_ref=int(n_ref), n_pred=int(n_pred), tp=tp, fp=int((~pred_hit[1:]).sum()), fn=n_ref - tp,
                case_ref=int(n_ref > 0), case_pred=int(n_pred > 0), count_error=abs(int(n_pred) - int(n_ref)))


def summarize(rows: Sequence[Dict]) -> Dict:
    """Pools the per-case counts of ``hits``. Pooled, not averaged: a case with 112 lesions weighs 112 times a case
    with one. That is the project's ranking measure; the per-subject average is reported separately because the two
    differ by up to 0.06 (ARBEIT.md 5j)."""
    total = {k: int(sum(r[k] for r in rows)) for k in ("n_ref", "n_pred", "tp", "fp", "fn")}
    n = len(rows)
    precision = total["tp"] / max(total["tp"] + total["fp"], 1)
    sensitivity = total["tp"] / max(total["tp"] + total["fn"], 1)
    case = dict(tp=sum(r["case_ref"] and r["case_pred"] for r in rows),
                fp=sum((not r["case_ref"]) and r["case_pred"] for r in rows),
                tn=sum((not r["case_ref"]) and (not r["case_pred"]) for r in rows),
                fn=sum(r["case_ref"] and (not r["case_pred"]) for r in rows))
    return dict(cases=n, **total, precision=round(precision, 4), sensitivity=round(sensitivity, 4),
                f1=round(2 * total["tp"] / max(2 * total["tp"] + total["fp"] + total["fn"], 1), 4),
                fp_per_case=round(total["fp"] / max(n, 1), 3),
                count_error_mean=round(float(np.mean([r["count_error"] for r in rows])), 3),
                case_level={k: int(v) for k, v in case.items()},
                case_accuracy=round((case["tp"] + case["tn"]) / max(n, 1), 4))


# --------------------------------------------------------------------------- VALDO rule
def centroids(labels: np.ndarray, n: int, spacing) -> np.ndarray:
    """Centres of mass of components 1..n in MILLIMETRES -> (n, 3)."""
    if n == 0:
        return np.zeros((0, 3))
    centres = np.asarray(ndimage.center_of_mass(labels > 0, labels, range(1, n + 1)), float).reshape(n, 3)
    return centres * np.asarray(spacing, float)


def match_greedy(reference_mm: np.ndarray, prediction_mm: np.ndarray,
                 distance_mm: float = VALDO_DISTANCE_MM) -> List[Tuple[int, int]]:
    """One-to-one match of reference to predicted centroids, greedy best-first.

    All pairs closer than ``distance_mm`` are sorted by distance (ties broken by reference index, then prediction
    index -- deterministic, so the number is reproducible) and accepted while neither side is taken. The challenge
    paper says "the one with the best association value is attributed" without giving an algorithm; greedy
    best-first is the reading this project uses, and it is stated rather than hidden. It is not guaranteed optimal
    (a Hungarian assignment could match one pair more in rare configurations).
    """
    if not len(reference_mm) or not len(prediction_mm):
        return []
    d = np.linalg.norm(reference_mm[:, None, :] - prediction_mm[None, :, :], axis=2)
    pairs = [(d[i, j], i, j) for i in range(d.shape[0]) for j in range(d.shape[1]) if d[i, j] < distance_mm]
    pairs.sort()
    ref_taken, pred_taken, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in ref_taken and j not in pred_taken:
            ref_taken.add(i); pred_taken.add(j); matched.append((i, j))
    return matched


def case_valdo(reference, probability, spacing, threshold: float = VALDO_THRESHOLD,
               distance_mm: float = VALDO_DISTANCE_MM) -> Dict:
    """One case under the official rule -> n_ref, n_pred, tp, fp, fn, f1 (nan without reference lesions), aed.

    Two deliberate differences from ``code/metric_valdo.py``, both verified not to change any reported number:
    f1 is returned UNROUNDED (the old file rounded to four decimals; rounding belongs in the reporting layer), and
    the diagnostic extras of the old file (``match_dist_mm_max``, ``multi_hit``, ``m26_*``) are not carried over --
    they are used only by that file's own command line and nowhere else in the project.
    """
    ref_labels, n_ref = ndimage.label(np.asarray(reference) > 0, structure=CONNECTIVITY_6)
    pred_labels, n_pred = ndimage.label(np.asarray(probability) > threshold, structure=CONNECTIVITY_6)
    matched = match_greedy(centroids(ref_labels, n_ref, spacing), centroids(pred_labels, n_pred, spacing), distance_mm)
    tp = len(matched)
    denominator = 2 * tp + (n_pred - tp) + (n_ref - tp)
    return dict(n_ref=int(n_ref), n_pred=int(n_pred), tp=tp, fp=int(n_pred - tp), fn=int(n_ref - tp),
                f1=(2 * tp / denominator) if n_ref > 0 and denominator else float("nan"),
                aed=abs(int(n_pred) - int(n_ref)))
