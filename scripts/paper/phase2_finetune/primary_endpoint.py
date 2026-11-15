"""wk0b PRIMARY endpoint from the stored results (CPU, no GPU, 06.11.2026):
chain F1 per n (stage 2 frozen-V(a)), the plateau n (per job spec: smallest n whose
paired case-bootstrap one-sided 95 % lower bound of F1(n) - F1(all) is > -0.02 and
stays so for every larger n), the descriptive fit F1(n) = a - b*n^(-c) (scipy
curve_fit, bounded multi-start) and the n at 95 % of the fitted plateau a.

Chain F1(n) = F1 of the pooled per-case (tp, fp, fn) counts over the 5 frozen test
folds and all seeds of the same draw (same test scans for every n -> paired).
Inputs (all stored, no retraining):
  phase4/chains_frozenV.json   stage-2 frozen-V(a) per block; per_case: {case: (tp, fp, fn)}
  paper/phase2_summary.json    reference chain F1 (wk2 S/P control)
Output: phase4/primary_endpoint.json
"""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json, re, warnings
import numpy as np

warnings.filterwarnings("ignore")
P4 = _P + "/paper/phase4"
N_ALL = {"t2star": 60, "swi": 49}
N_ORDER = [3, 5, 10, 15, 20, 30, 40]
RNG = np.random.default_rng(42)
N_BOOT = 2000

def f1_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

def chain_by_n(mod, variant):
    """(n) -> {case: [tp, fp, fn]} pooled over folds and all seeds."""
    d = json.load(open(f"{P4}/chains_frozenV.json"))
    out = {}
    for k, v in d.items():
        m = re.match(rf"{mod}/({variant})_n(\d+|all)_fold(\d)_s(\d+)$", k)
        if not m:
            continue
        n = N_ALL[mod] if m.group(2) == "all" else int(m.group(2))
        acc = out.setdefault(n, {})
        for c, (tp_, fp_, fn_) in v["a_pooled"]["per_case"].items():
            r = acc.setdefault(c, [0, 0, 0])
            r[0] += tp_; r[1] += fp_; r[2] += fn_
    return out

def paired_case_lb(fN, fALL, n_boot=N_BOOT):
    """one-sided 95 % lower bound of F1(n) - F1(all), paired bootstrap of the cases."""
    cases = [c for c in fALL if c in fN]
    a = np.array([fN[c] for c in cases], float)
    b = np.array([fALL[c] for c in cases], float)
    idx = RNG.integers(0, len(cases), size=(n_boot, len(cases)))
    A = a[idx].sum(axis=1); B = b[idx].sum(axis=1)
    diffs = 2 * A[:, 0] / np.maximum(2 * A[:, 0] + A[:, 1] + A[:, 2], 1) - \
            2 * B[:, 0] / np.maximum(2 * B[:, 0] + B[:, 1] + B[:, 2], 1)
    return float(np.percentile(diffs, 2.5)), len(cases)

def fit_curve(xs, ys):
    from scipy.optimize import curve_fit
    best, best_sse = None, None
    for a0 in np.linspace(0.5, 0.95, 9):
        for b0 in np.linspace(0.05, 0.5, 5):
            for c0 in np.linspace(0.2, 2.0, 7):
                try:
                    p, _ = curve_fit(lambda x, a, b, c: a - b * x ** (-c), xs, ys,
                                     p0=[a0, b0, c0], maxfev=4000,
                                     bounds=([0.1, 1e-3, 0.05], [1.5, 3.0, 3.0]))
                except Exception:
                    continue
                if not (np.all(np.isfinite(p)) and p[0] > 0 and p[1] > 0):
                    continue
                sse = float(np.sum((p[0] - p[1] * xs ** (-p[2]) - ys) ** 2))
                if best_sse is None or sse < best_sse:
                    best, best_sse = p, sse
    if best is None:
        return None
    a_, b_, c_ = [float(x) for x in best]
    n95 = (b_ / (0.05 * a_)) ** (1 / c_) if a_ > 0 else None
    return {"a": a_, "b": b_, "c": c_, "n_at_95pct_of_a": round(n95, 1) if n95 else None,
            "note": "descriptive fit on n = 3..40; n_at_95pct_of_a = analytic solution of a - b*n^(-c) = 0.95*a"}

def main():
    ref = json.load(open(_P + "/paper/phase2_summary.json"))
    ref_f1 = {m: ref["size_matched"]["size_and_diversity_per_dataset"][m]["F1_SP"] for m in ("swi", "t2star")}
    out = {
        "method": ("chain F1(n) = stage-2 frozen-V(a), per-case counts pooled over the 5 frozen test folds and all seeds of the same draw "
                   "(same test scans for every n -> paired). Plateau n = smallest n whose paired case-bootstrap one-sided 95 % lower bound of "
                   "F1(n) - F1(all) (> -0.02, 2000 resamples) holds at n and at every larger n. Fit F1(n) = a - b*n^(-c) (scipy curve_fit, "
                   "bounded multi-start, n = 3..40 only). Reference chain F1 = wk2 S/P control (in-site, all scans). "
                   "FTn has no n = all by design (spec: FTn only at n = 5, 10, 20) -> its plateau n is not defined."),
        "reference_chain_f1": ref_f1, "n_all": N_ALL, "targets": {}}
    for mod in ("t2star", "swi"):
        tgt = {"chain": {}}
        for variant in ("S", "FT", "FTn"):
            chain = chain_by_n(mod, variant)
            if not chain:
                continue
            curve = {}
            for n in sorted(chain):
                tot = np.sum(list(chain[n].values()), axis=0)
                curve[n] = {"f1": f1_counts(*tot), "tp": int(tot[0]), "fp": int(tot[1]), "fn": int(tot[2]),
                            "n_cases": len(chain[n])}
            if N_ALL[mod] in curve:
                per_n = {}
                for n in N_ORDER + [N_ALL[mod]]:
                    if n in curve:
                        lb, nc = paired_case_lb(chain[n], chain[N_ALL[mod]])
                        per_n[str(n)] = {"f1": curve[n]["f1"], "lb95_F1n_minus_F1all": lb, "n_cases_paired": nc}
                plateau_n = None
                for n in N_ORDER + [N_ALL[mod]]:
                    if n not in per_n:
                        continue
                    if all(per_n[str(m)]["lb95_F1n_minus_F1all"] > -0.02
                           for m in N_ORDER + [N_ALL[mod]] if m >= n and m in per_n):
                        plateau_n = n
                        break
                fit = None
                try:
                    ns = [n for n in N_ORDER if n in curve]
                    fit = fit_curve(np.array(ns, float), np.array([curve[n]["f1"] for n in ns]))
                except Exception as e:
                    fit = {"error": f"{type(e).__name__}: {e}"}
                tgt["chain"][variant] = {"curve": {str(k): v for k, v in sorted(curve.items())},
                                         "per_n": per_n, "plateau_n": plateau_n, "fit": fit,
                                         "f1_all": curve[N_ALL[mod]]["f1"]}
            else:
                tgt["chain"][variant] = {"status": "no n = all (by design for FTn)",
                                         "curve": {str(k): v for k, v in sorted(curve.items())},
                                         "plateau_n": None, "fit": None}
        out["targets"][mod] = tgt
    json.dump(out, open(f"{P4}/primary_endpoint.json", "w"), indent=1)
    print("written", f"{P4}/primary_endpoint.json")
    for mod in ("t2star", "swi"):
        for v in ("S", "FT", "FTn"):
            c = out["targets"][mod]["chain"].get(v, {})
            if c.get("fit") is None and "plateau_n" not in c:
                print(f"{mod} {v}: {c.get('status')}")
            else:
                f = c.get("fit") or {}
                print(f"{mod} {v}: plateau_n={c.get('plateau_n')}  F1(all)={c.get('f1_all', 0):.3f}  "
                      f"a={f.get('a', 0):.3f} n95(a)={f.get('n_at_95pct_of_a')}")

if __name__ == "__main__":
    main()
