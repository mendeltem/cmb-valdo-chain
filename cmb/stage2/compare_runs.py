"""Paired comparison of two second-stage results that were computed on DIFFERENT candidate sets.

cmb.stage2.pooled compares its modes on one candidate set. To ask "does a sensitive first stage plus the shared classifier
beat the baseline first stage with its simple rule?" the two sides come from different runs of that script. Both JSON files
carry per-case counts (tp, fp, fn); cases are the same, so a bootstrap over cases gives the paired interval.

    python -m cmb.stage2.compare_runs --a FILE_A.json "MODE A" --b FILE_B.json "MODE B" [--cohorts valdo t2star swi]
"""
import argparse
import json

import numpy as np


def f1(rows) -> float:
    tp, fp, fn = (sum(r[i] for r in rows) for i in range(3))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs=2, required=True, metavar=("JSON", "MODE"))
    parser.add_argument("--b", nargs=2, required=True, metavar=("JSON", "MODE"))
    parser.add_argument("--cohorts", nargs="*")
    args = parser.parse_args()
    a = json.load(open(args.a[0]))["per_case"][args.a[1]]; b = json.load(open(args.b[0]))["per_case"][args.b[1]]
    rng = np.random.RandomState(0)
    for cohort in args.cohorts or sorted(set(a) & set(b)):
        ids = sorted(set(a[cohort]) & set(b[cohort])); A = [a[cohort][i] for i in ids]; B = [b[cohort][i] for i in ids]
        draws = []
        for _ in range(10000):
            ix = rng.randint(len(ids), size=len(ids)); draws.append(f1([A[j] for j in ix]) - f1([B[j] for j in ix]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"{cohort:8s} n={len(ids):3d}  A {f1(A):.3f}  B {f1(B):.3f}  A - B {f1(A) - f1(B):+.3f} [{lo:+.3f}; {hi:+.3f}]")


if __name__ == "__main__":
    main()
