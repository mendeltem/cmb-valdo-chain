"""MITTEL #2 (06.11., Gutachter): draw-strategy arms NONE/LOW/MID/HIGH vs STRAT (FT, n=5/10/20, 3 seeds each).
Chain F1 pooled over folds+seeds for the same (n, strategy); report F1 vs n AND vs the number of annotated
CMB in the n drawn training scans (annotation cost scales with CMB count). CPU, from stored per-case counts.
Output: phase4/draw_strategies.json
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
import json, re, glob, collections
import numpy as np

P4 = _P + "/paper/phase4"
N_ALL = {"t2star": 60, "swi": 49}

def f1c(tp, fp, fn):
    return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0

d = json.load(open(f"{P4}/chains_frozenV.json"))
STRATS = ["STRAT", "NONE", "LOW", "MID", "HIGH"]
out = {"note": "Draw-strategy arms, FT stage-1, frozen-V(a) stage-2 chain F1. STRAT is the default (lesion-count stratified) used for the headline curve. NONE=CMB-free scans only, LOW=feast CMB-first (>=1 CMB), MID=closest to median, HIGH=most CMB. n_cmb = total annotated CMB in the n drawn training scans (annotation cost). Pooled over the 5 frozen folds and the seeds of the same draw (paired).",
       "targets": {}}
for mod in ("t2star", "swi"):
    # gather per (n, strategy): accumulate per-case tp/fp/fn
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0,0,0]))
    cumb = {}   # (n,strategy) -> list of n_cmb over folds/seeds
    for k, v in d.items():
        m = re.match(rf"{mod}/(FT)_n(\d+)_fold(\d)_s(\d+)(_STRAT|_NONE|_LOW|_MID|_HIGH)?$", k)
        if not m or "a_pooled" not in v:
            continue
        V, n, fl, s, sfx = m.groups(); sfx = (sfx or "_STRAT").lstrip("_")
        key = (int(n), sfx)
        for c,(tp,fp,fn) in v["a_pooled"]["per_case"].items():
            r = acc[key].setdefault(c, [0,0,0]); r[0]+=tp; r[1]+=fp; r[2]+=fn
        # read n_cmb from the draw file for this fold+seed+strategy
        df = f"{P4}/draws/{mod}/fold{fl}_n{n}_{sfx}_{s}.json"
        try:
            nd = json.load(open(df)); cumb.setdefault(key, []).append(int(nd.get("n_cmb", -1)))
        except Exception:
            cumb.setdefault(key, []).append(-1)
    tgt = {}
    for n in (5, 10, 20):
        row = {}
        for st in STRATS:
            if (n, st) not in acc or not acc[(n, st)]:
                row[st] = None; continue
            cases = acc[(n, st)]
            tp = np.array([r[0] for r in cases.values()], float)
            fp = np.array([r[1] for r in cases.values()], float)
            fn = np.array([r[2] for r in cases.values()], float)
            tot = [int(tp.sum()), int(fp.sum()), int(fn.sum())]
            cumb_here = cumb.get((n, st), [])
            good = [c for c in cumb_here if c >= 0]
            row[st] = {
                "f1": f1c(*tot), "tp": tot[0], "fp": tot[1], "fn": tot[2],
                "n_cases": len(cases),
                "n_cmb_annotated": int(np.mean(good)) if good else None,   # avg CMB annotated across the draws
                "fp_per_scan": round(tot[1] / max(len(cases),1), 3),
            }
        tgt[str(n)] = row
    out["targets"][mod] = tgt

json.dump(out, open(f"{P4}/draw_strategies.json", "w"), indent=1)
print("written", f"{P4}/draw_strategies.json")
for mod in ("t2star","swi"):
    print(f"\n== {mod} (F1 / n_cmb annotated / FP-per-scan)")
    for n in ("5","10","20"):
        print(f"  n={n}:")
        for st in STRATS:
            r = out["targets"][mod][n].get(st)
            if r is None:
                print(f"    {st:5s}: (no block)"); continue
            print(f"    {st:5s}: F1={r['f1']:.3f}  n_cmb~{r['n_cmb_annotated']}  fp/scan={r['fp_per_scan']}")
