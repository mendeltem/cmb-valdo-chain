"""wk9 STEP 2: comparison of the frozen A1v2 chain with the frozen-V chain.

Per modality x variant (S/FT/FTn, threshold-only variant (a)) x n:
  lesion F1 (pooled over folds and all seeds of the same draw) and its change
  (A1v2 - frozenV), plus recall by size class (<2, 2-5, >5 mm).

The size-class recall needs the per-block target candidate sets (lesion diameter
in mm), i.e. phase4/candidates -- with cmb.analysis.size_stratified semantics
(reference lesion mm size from the candidate rows, hits = per-case tp restricted
to lesions of the class). If the candidates are not available, the size section
is written as {"size": {"status": "pending: ..."}} -- an honest state, not a
placeholder number.

Output: phase4/a1v2_vs_frozenV.json
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

import numpy as np

P4 = _P + "/paper/phase4"
N_ALL = {"t2star": 60, "swi": 49}
N_ORDER = [3, 5, 10, 15, 20, 30, 40, "all"]
SIZE_CLASSES = {"<2mm": (0.0, 2.0), "2-5mm": (2.0, 5.0), ">5mm": (5.0, np.inf)}


def f1_counts(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def load_chain(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def chain_by_n(d, mod, variant):
    """(n) -> {case: [tp, fp, fn]} pooled over folds and all seeds (variant (a))."""
    out = {}
    for k, v in d.items():
        m = re.match(rf"{mod}/({variant})_n(\d+|all)_fold(\d)_s(\d+)$", k)
        if not m or "error" in v or "a_pooled" not in v:
            continue
        n = N_ALL[mod] if m.group(2) == "all" else int(m.group(2))
        acc = out.setdefault(n, {})
        for c, (tp_, fp_, fn_) in v["a_pooled"]["per_case"].items():
            r = acc.setdefault(c, [0, 0, 0])
            r[0] += tp_; r[1] += fp_; r[2] += fn_
    return out


def size_recall(mod):
    """recall by size class (<2, 2-5, >5 mm) for one modality, pooled over the whole
    frozen-A1v2 chain (winner tset-plain = frozen-V), variant (a) threshold.

    The frozen-A1v2 classifier is VALDO-only and identical for every n (no fine-tuning),
    so the size-class recall is n-independent and the same for every (variant, n) row.
    Pre-computed with cmb.analysis.size_stratified on the block decision dumps
    (a1v2_decisions_sizedump.jsonl) + the gitter label files; stored in a1v2_size_stratified.json.
    """
    try:
        d = json.load(open(f"{P4}/a1v2_size_stratified.json"))
        coh = d["cohorts"].get(mod)
        if not coh:
            return None
        mode = list(next(iter(coh.values())).get("chain_recall", {}).keys())[0] if any(
            isinstance(v, dict) and v.get("chain_recall") for v in coh.values()) else None
        if mode is None:
            return None
        out = {}
        for cls in ("<2 mm", "2-5 mm", ">5 mm", "all"):
            v = coh.get(cls, {})
            out[cls] = {
                "n": v.get("n"),
                "share": v.get("share"),
                "stage1_ceiling_recall": v.get("stage1_ceiling_recall"),
                "chain_recall": v.get("chain_recall", {}).get(mode),
            }
        out["median_diameter_mm"] = coh.get("median_diameter_mm")
        out["definition"] = d.get("definition", "")
        return out
    except (FileNotFoundError, KeyError, IndexError, StopIteration):
        return None


def main():
    frozenV = load_chain(f"{P4}/chains_frozenV.json")
    a1v2 = load_chain(f"{P4}/chains_frozenA1v2.json")
    try:
        sel = json.load(open(_P + "/stage2/a1v2/selection.json"))
        winner_arm = sel["winner"]
    except Exception:
        sel = {}; winner_arm = "pending: selection.json not written yet (seed runs in progress)"
    out = {
        "method": "chain F1(n) per (mod, variant, n) = pooled per-case counts over the 5 frozen test "
                  "folds and all seeds of the same draw, variant (a) threshold only. A1v2 = the "
                  "VALDO-only winner of MB/stage2/a1v2/selection.json applied unchanged; "
                  "frozenV = the paper chain (tset-plain, variant (a)). size = recall by lesion "
                  "size class (<2, 2-5, >5 mm; cmb.analysis.size_stratified).",
        "winner_arm": winner_arm,
        "frozenV_source": f"{P4}/chains_frozenV.json",
        "a1v2_source": f"{P4}/chains_frozenA1v2.json",
        "per_block": {},
    }
    if a1v2 is None:
        out["a1v2_status"] = "pending: chains_frozenA1v2.json not produced this round"
    else:
        n_err = sum(1 for v in a1v2.values() if "error" in v)
        out["a1v2_blocks"] = len(a1v2)
        out["a1v2_error_blocks"] = n_err
        if n_err == len(a1v2):
            out["a1v2_status"] = "pending: all blocks error (target candidate sets not locally " \
                                 "available; see ARBEIT.md wk9)"
    for mod in ("swi", "t2star"):
        out["per_block"][mod] = {}
        for variant in ("S", "FT", "FTn"):
            cV = chain_by_n(frozenV, mod, variant) if frozenV else {}
            cA = chain_by_n(a1v2, mod, variant) if a1v2 else {}
            ns = [n for n in N_ORDER if n in cV]
            if variant == "FTn":
                ns = [n for n in ns if n in (5, 10, 20)]
            table = {}
            for n in ns:
                tp = sum(t[0] for t in cV[n].values()); fp = sum(t[1] for t in cV[n].values())
                fn = sum(t[2] for t in cV[n].values())
                fV = f1_counts(tp, fp, fn)
                row = {"n": n, "frozenV_f1": round(fV, 4), "frozenV_cases": len(cV[n])}
                if n in cA:
                    tpA = sum(t[0] for t in cA[n].values()); fpA = sum(t[1] for t in cA[n].values())
                    fnA = sum(t[2] for t in cA[n].values())
                    fA = f1_counts(tpA, fpA, fnA)
                    row["a1v2_f1"] = round(fA, 4)
                    row["delta_f1"] = round(fA - fV, 4)
                    row["size"] = size_recall(mod) or \
                        {"status": "pending: size recall not computed this round"}
                else:
                    row["a1v2_f1"] = None
                    row["size"] = {"status": "pending: a1v2 chain not computed this round"}
                table[str(n)] = row
            out["per_block"][mod][variant] = table
    json.dump(out, open(f"{P4}/a1v2_vs_frozenV.json", "w"), indent=1)
    print("written to", f"{P4}/a1v2_vs_frozenV.json")


if __name__ == "__main__":
    main()
