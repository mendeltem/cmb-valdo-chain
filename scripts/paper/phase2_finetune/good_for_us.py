"""wk0b SECONDARY 'good for us' n (CPU, 06.11.2026), from the stored predictions only
(no new training). Per target and variant: the smallest n whose chain (frozen-V(a))
reaches ALL of:
  (1) F1 >= 95 % of the reference chain F1 of D (wk2 S/P, MB/paper/phase2_summary.json);
  (2) F1 not below the best external baseline of wk1 on D: one-sided 95 % lower bound of
      F1 - baseline > -0.02 (case bootstrap 2000, MB/extern/summary.json, v2 nested values);
  (3) subject-level count ICC(3,1) >= 0.75 (two-way mixed, consistency, single measure;
      cmb.agreement.icc31. The paper version's own copy returned ~0.997 for any data, fixed 16.11.2026).
Chain F1(n) and the per-case (ref, pred) counts come from phase4/chains_frozenV.json
(fold-pooled, all seeds of the same draw). Output: phase4/good_for_us.json
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
import json
import numpy as np
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))  # repo root, for cmb.agreement
from cmb.agreement import icc31  # ICC(3,1), Shrout & Fleiss 1979 (own copy was wrong, fixed 16.11.2026)

P4 = _P + "/paper/phase4"
N_ALL = {"t2star": 60, "swi": 49}
N_ORDER = [3, 5, 10, 15, 20, 30, 40]
RNG = np.random.default_rng(42)
N_BOOT = 2000

REF_F1 = {"swi": 0.6649, "t2star": 0.6672}                       # wk2 S/P (phase2_summary.json)
EXT_BEST = {"swi": {"f1": 0.637, "model": "SHIVA-CMB v2 nested"},
            "t2star": {"f1": 0.424, "model": "SHIVA-CMB v2 nested"}}   # wk1b, best external on D

def f1_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def main():
    d = json.load(open(f"{P4}/chains_frozenV.json"))
    import re
    out = {"criteria": "the smallest n whose chain reaches (1) F1 >= 95 % of the reference chain F1 of D (wk2 S/P), "
                       "AND (2) one-sided 95 % lower bound (case bootstrap, 2000) of F1 - best wk1 external baseline > -0.02, "
                       "AND (3) ICC(3,1) of the subject-level counts >= 0.75. Chain = stage-2 frozen-V(a), folds pooled, all seeds of the same draw.",
           "reference_f1": REF_F1, "ext_best": EXT_BEST, "targets": {}}
    for mod in ("t2star", "swi"):
        chain = {}
        for k, v in d.items():
            m = re.match(rf"{mod}/(S|FT|FTn)_n(\d+|all)_fold(\d)_s(\d+)$", k)
            if not m:
                continue
            n = N_ALL[mod] if m.group(2) == "all" else int(m.group(2))
            acc = chain.setdefault((m.group(1), n), {})
            for c, (tp_, fp_, fn_) in v["a_pooled"]["per_case"].items():
                r = acc.setdefault(c, [0, 0, 0])
                r[0] += tp_; r[1] += fp_; r[2] += fn_
        tgt = {}
        for variant in ("S", "FT", "FTn"):
            rows = {}
            for n in N_ORDER + [N_ALL[mod]]:
                if (variant, n) not in chain:
                    continue
                acc = chain[(variant, n)]
                tp = np.array([r[0] for r in acc.values()], float)
                fp = np.array([r[1] for r in acc.values()], float)
                fn = np.array([r[2] for r in acc.values()], float)
                ref = tp + fn; pred = tp + fp
                tot = np.sum(list(acc.values()), axis=0)
                f1 = f1_counts(*tot)
                idx = RNG.integers(0, len(ref), size=(N_BOOT, len(ref)))
                a, b, c = tp[idx], fp[idx], fn[idx]
                f1b = 2 * a.sum(axis=1) / np.maximum(2 * a.sum(axis=1) + b.sum(axis=1) + c.sum(axis=1), 1)
                icc = icc31(ref, pred)
                rows[str(n)] = {"f1": f1, "f1_ci95": [float(np.percentile(f1b, 2.5)), float(np.percentile(f1b, 97.5))],
                                "lb95_F1_minus_ext": float(np.percentile(f1b - EXT_BEST[mod]["f1"], 2.5)),
                                "icc31": round(icc, 4) if icc == icc else None,
                                "c1": bool(f1 >= 0.95 * REF_F1[mod]),
                                "c2": bool(np.percentile(f1b - EXT_BEST[mod]["f1"], 2.5) > -0.02),
                                "c3": bool(icc == icc and icc >= 0.75)}
            good = next((n for n in N_ORDER + [N_ALL[mod]] if str(n) in rows and
                         all(rows[str(n)][f"c{i}"] for i in (1, 2, 3))), None)
            tgt[variant] = {"per_n": rows, "good_for_us_n": good}
        out["targets"][mod] = tgt
    json.dump(out, open(f"{P4}/good_for_us.json", "w"), indent=1)
    print("written", f"{P4}/good_for_us.json")
    for mod in ("t2star", "swi"):
        for v in ("S", "FT", "FTn"):
            t = out["targets"][mod][v]
            print(f"{mod} {v}: good_for_us_n={t['good_for_us_n']}")
            for n, r in t["per_n"].items():
                print(f"   n={n:>4s} f1={r['f1']:.3f} ci=[{r['f1_ci95'][0]:.3f},{r['f1_ci95'][1]:.3f}] "
                      f"lb_ext={r['lb95_F1_minus_ext']:+.3f} icc={r['icc31']}  "
                      f"c1={int(r['c1'])} c2={int(r['c2'])} c3={int(r['c3'])}")

if __name__ == "__main__":
    main()
