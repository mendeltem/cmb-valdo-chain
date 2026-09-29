"""Checks that `cmb.metrics` reproduces the legacy versions `code/metric.py` and `code/metric_valdo.py` EXACTLY.

Acceptance rule of the package (cmb/README.md): a module only counts once it reproduces its predecessor exactly. Here
that means: the same counts on random masks AND on real data, and the same matching under the VALDO rule.

    python -m cmb.tests.test_metrics_match_legacy

Returns 0 if everything matches, 1 at the first deviation (with the case that triggers it).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/code")

import metric as alt_touch                    # noqa: E402
import metric_valdo as alt_valdo              # noqa: E402
from cmb import metrics as neu                # noqa: E402


def zufallsmasken(rng, form=(40, 50, 50), n=12):
    """Masks with realistic structure: a few small clumps, not noise -- otherwise everything falls apart into
    thousands of components and the comparison only tests the counting loop anymore."""
    for _ in range(n):
        a = np.zeros(form, bool); b = np.zeros(form, bool)
        for ziel, k in ((a, rng.integers(0, 6)), (b, rng.integers(0, 8))):
            for _ in range(k):
                z, y, x = (rng.integers(3, s - 4) for s in form)
                r = int(rng.integers(1, 3))
                ziel[z - r:z + r + 1, y - r:y + r + 1, x - r:x + r + 1] = True
        yield a, b


def f1_gleich(alt, neu_wert) -> bool:
    """Compares f1 tolerantly against two deliberate differences: the legacy version rounds to four digits, and for
    cases WITHOUT a reference microbleed it returns None instead of nan (both mean 'undefined')."""
    import math
    undefiniert = lambda v: v is None or (isinstance(v, float) and math.isnan(v))
    if undefiniert(alt) or undefiniert(neu_wert):
        return undefiniert(alt) and undefiniert(neu_wert)
    return round(float(neu_wert), 4) == round(float(alt), 4)


def main() -> int:
    fehler = 0
    rng = np.random.default_rng(0)

    # --- 1) touch rule on random masks
    for i, (ref, pred) in enumerate(zufallsmasken(rng, n=40)):
        a, b = alt_touch.hits(ref, pred), neu.hits(ref, pred)
        if a != b:
            print(f"DEVIATION hits, random case {i}: old {a} vs new {b}"); fehler += 1; break
    else:
        print("hits on 40 random masks: identical")

    # --- 2) summarize on the same rows
    zeilen = [neu.hits(r, p) for r, p in zufallsmasken(np.random.default_rng(1), n=25)]
    a, b = alt_touch.summarize(zeilen), neu.summarize(zeilen)
    if a != b:
        print(f"DEVIATION summarize: old {a} vs new {b}"); fehler += 1
    else:
        print("summarize on 25 cases: identical")

    # --- 3) VALDO rule: matching and counting
    for i, (ref, pred) in enumerate(zufallsmasken(np.random.default_rng(2), n=30)):
        a = alt_valdo.case_valdo(ref, pred.astype(np.float32), (0.5, 0.5, 1.0))
        b = neu.case_valdo(ref, pred.astype(np.float32), (0.5, 0.5, 1.0))
        # The legacy version rounds f1 to four digits and returns extra fields (match_dist_mm_max, multi_hit, m26_*),
        # which only its own command-line program uses. The shared fields are compared; f1 with
        # the same rounding -- the new version deliberately does NOT round, that belongs in the reporting layer.
        gleich = all(a[k] == b[k] for k in ("n_ref", "n_pred", "tp", "fp", "fn", "aed")) and f1_gleich(a["f1"], b["f1"])
        if not gleich:
            print(f"DEVIATION case_valdo, random case {i}: old {a} vs new {b}"); fehler += 1; break
    else:
        print("case_valdo on 30 random masks: identical")

    # --- 4) real data: reference against a real prediction
    gitter, lauf = f"{ROOT}/dev/gitter_d", f"{ROOT}/ergebnisse/turnier/a03-aniso-e60-gd"
    pfade = sorted(glob.glob(f"{lauf}/fold[0-9]/*_pred_proba.nii.gz"))[:8]
    if not pfade:
        print("real data: skipped (no predictions available)")
    else:
        import nibabel as nib
        n_gleich = 0
        for pfad in pfade:
            fall = os.path.basename(pfad)[: -len("_pred_proba.nii.gz")]
            p = np.asarray(nib.load(pfad).dataobj).astype(np.float32) / 255.0
            ref = np.asarray(nib.load(f"{gitter}/{fall}/{fall}_label.nii.gz").dataobj) > 0
            if alt_touch.hits(ref, p > 0.3) != neu.hits(ref, p > 0.3):
                print(f"DEVIATION hits on real case {fall}"); fehler += 1; break
            a, b = alt_valdo.case_valdo(ref, p, (0.5, 0.5, 1.0)), neu.case_valdo(ref, p, (0.5, 0.5, 1.0))
            if not all(a[k] == b[k] for k in ("tp", "fp", "fn", "aed")):
                print(f"DEVIATION case_valdo on real case {fall}: old {a} vs new {b}"); fehler += 1; break
            n_gleich += 1
        else:
            print(f"real data: {n_gleich} cases, both rules identical")

    print("\nall identical" if not fehler else f"\n{fehler} deviation(s)")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
