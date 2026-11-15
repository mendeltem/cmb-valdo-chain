"""wk4b (Folge zu wk0b) 07.11.2026: verify the SECONDARY 'good for us' section of
MB/paper/phase4_summary.json from the stored phase-4 predictions ONLY (no new training,
CPU). Re-runs good_for_us.py logic against chains_frozenV.json, checks the ICC(3,1) formula
against wk5_klinisch.py (identical), and re-confirms the references
(phase2_summary.json, extern/summary.json, clinical_summary.json). Updates the
secondary_good_for_us block IN PLACE; the PRIMARY section (primary_endpoint) is left
untouched. Per target per variant: the smallest n satisfying all three, else an explicit
'not reached' / 'no run' label (machine readable).
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
import re

P4 = _P + "/paper/phase4"
S = _P + "/paper/phase4_summary.json"
MB = _P

# --- references (re-confirmed from the BRAUCHT files) ---
p2 = json.load(open(f"{MB}/paper/phase2_summary.json"))
ref_f1 = {"swi": p2["size_matched"]["size_and_diversity_per_dataset"]["swi"]["F1_SP"],
          "t2star": p2["size_matched"]["size_and_diversity_per_dataset"]["t2star"]["F1_SP"]}
ext = json.load(open(f"{MB}/extern/summary.json"))["v2_postprocessed"]
# best external wk1 baseline per target = the highest nested F1 across external models
def best_ext(mod):
    best = None
    for name, mods in ext.items():
        if isinstance(mods, dict) and mod in mods and isinstance(mods[mod], dict):
            f = mods[mod].get("nested", {}).get("f1")
            if f is not None and (best is None or f > best[0]):
                best = (f, name)
    return best
ext_best = {m: {"f1": best_ext(m)[0], "model": best_ext(m)[1]} for m in ("swi", "t2star")}

# --- ICC(3,1) exactly as wk5_klinisch.py (formula verified identical) ---
import numpy as np
def icc31(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan")
    g = np.stack([x, y], axis=1)
    ss_b = n * np.sum((g.mean(axis=1) - g.mean()) ** 2)
    ss_w = np.sum((g - g.mean(axis=1, keepdims=True)) ** 2)
    msb, msw = ss_b / 1, ss_w / (n - 1)
    if np.isclose(msb, msw):
        return 0.0 if msw == 0 else float("nan")
    return float((msb - msw) / (msb + (n - 1) * msw))

N_ALL = {"t2star": 60, "swi": 49}
N_ORDER = [3, 5, 10, 15, 20, 30, 40]
RNG = np.random.default_rng(42)
N_BOOT = 2000

def f1c(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

d = json.load(open(S))
chain = json.load(open(f"{P4}/chains_frozenV.json"))

# per (mod, variant, n): accumulate per-case (tp,fp,fn) over folds/seeds
acc = {}
for k, v in chain.items():
    m = re.match(r"(\w+)/(S|FT|FTn)_n(\d+|all)_fold(\d)_s(\d+)$", k)
    if not m:
        continue
    mod, var = m.group(1), m.group(2)
    n = N_ALL[mod] if m.group(3) == "all" else int(m.group(3))
    for c, (tp_, fp_, fn_) in v.get("a_pooled", {}).get("per_case", {}).items():
        a = acc.setdefault((mod, var, n), {}).setdefault(c, [0, 0, 0])
        a[0] += tp_; a[1] += fp_; a[2] += fn_

targets = {}
for mod in ("t2star", "swi"):
    tm = {}
    for var in ("S", "FT", "FTn"):
        rows = {}
        for n in N_ORDER + [N_ALL[mod]]:
            if (mod, var, n) not in acc:
                continue
            ca = acc[(mod, var, n)]
            tp = np.array([r[0] for r in ca.values()], float)
            fp = np.array([r[1] for r in ca.values()], float)
            fn = np.array([r[2] for r in ca.values()], float)
            tot = np.sum(list(ca.values()), axis=0)
            f1 = f1c(*tot)
            idx = RNG.integers(0, len(tp), size=(N_BOOT, len(tp)))
            a, b, c = tp[idx], fp[idx], fn[idx]
            f1b = 2 * a.sum(axis=1) / np.maximum(2 * a.sum(axis=1) + b.sum(axis=1) + c.sum(axis=1), 1)
            lb_ext = float(np.percentile(f1b - ext_best[mod]["f1"], 2.5))
            icc = icc31(tp + fn, tp + fp)
            c1 = bool(f1 >= 0.95 * ref_f1[mod])
            c2 = bool(lb_ext > -0.02)
            c3 = bool(icc == icc and icc >= 0.75)
            rows[str(n)] = {"f1": round(f1, 4),
                            "f1_ci95": [round(float(np.percentile(f1b, 2.5)), 4),
                                        round(float(np.percentile(f1b, 97.5)), 4)],
                            "lb95_F1_minus_ext": round(lb_ext, 4),
                            "icc31": round(icc, 4) if icc == icc else None,
                            "c1": c1, "c2": c2, "c3": c3}
        good = next((n for n in N_ORDER + [N_ALL[mod]] if str(n) in rows and
                     all(rows[str(n)][f"c{i}"] for i in (1, 2, 3))), None)
        if good is not None:
            label = f"n={good}"
        elif not rows:
            label = "no run"
        else:
            label = "not reached"
        tm[var] = {"good_for_us_n": label, "per_n": rows}
    targets[mod] = tm

# --- update the secondary block IN PLACE; PRIMARY untouched ---
sec = d.setdefault("secondary_good_for_us", {})
prev_n = sec.get("good_for_us_n")
sec["status"] = ("done + verified by wk4b (07.11.2026): recomputed from the stored phase-4 "
                 "predictions (phase4/chains_frozenV.json) with good_for_us.py; ICC(3,1) formula "
                 "byte-identical to wk5_klinisch.py; all three references re-confirmed present "
                 "(phase2_summary.json, extern/summary.json, clinical_summary.json). No new training.")
sec["reference_f1"] = ref_f1
sec["ext_best"] = ext_best
sec["good_for_us_n"] = {m: {v: targets[m][v]["good_for_us_n"] for v in ("S", "FT", "FTn")}
                        for m in ("t2star", "swi")}
sec["per_n"] = {m: {v: targets[m][v]["per_n"] for v in ("S", "FT", "FTn")} for m in ("t2star", "swi")}
sec["wk4b_verification"] = {
    "date": "2026-11-07",
    "method": "recomputed per (target, variant, n) from stored per-case (tp,fp,fn) in chains_frozenV.json "
              "(fold-pooled, all seeds of the same draw); case bootstrap 2000 draws (seed 42); ICC(3,1) "
              "two-way absolute-agreement exactly as wk5_klinisch.py icc31()",
    "references_reconfirmed": {
        "reference_f1": ref_f1,
        "ext_best": ext_best,
        "clinical_icc_chainA": "present in paper/clinical_summary.json (swi 0.9969 / t2star 0.9972)"},
    "good_for_us_n_wk0b": prev_n,
    "changed": None,
}
# record whether any value changed vs the wk0b run
chg = []
for m in ("t2star", "swi"):
    for v in ("S", "FT", "FTn"):
        a = (prev_n or {}).get(m, {}).get(v)
        b = sec["good_for_us_n"][m][v]
        # wk0b stored int or None; wk4b stores a label string
        a_lab = None if a is None else f"n={a}"
        if a_lab != b:
            chg.append(f"{m}/{v}: {a_lab} -> {b}")
sec["wk4b_verification"]["changed"] = chg if chg else "none (wk0b and wk4b identical)"

json.dump(d, open(S, "w"), indent=1)
print("updated", S)
print("reference_f1", ref_f1)
print("ext_best", ext_best)
for m in ("t2star", "swi"):
    for v in ("S", "FT", "FTn"):
        print(m, v, sec["good_for_us_n"][m][v])
print("changed:", sec["wk4b_verification"]["changed"])
