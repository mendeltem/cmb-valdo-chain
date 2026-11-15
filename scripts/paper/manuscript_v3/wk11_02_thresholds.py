"""wk11b step 2 of 6 (10.11.2026): threshold recalibration -- one set of numbers for 3.2, 3.3 and 4.2.

CPU only, from result files already on disk. Reads no result file fully into a report: prints aggregates only.

What is computed, on all five folds, for T2*GRE and SWI:
  (a) cmb_free       -- threshold and minimum size re-chosen on n CMB-free training scans (no annotation at all)
  (b) annotated      -- the same re-chosen on n affected training scans
  (c) upper_bound    -- the best cell per fold chosen on that fold's TEST scans (not attainable, upper bound only)
plus the n = 0 point (the model as downloaded, fixed 0.3 / 2 mm3) and, per fold, whether (a) beats n = 0.

Sources (aggregate keys only, never the *_blocks.json files in full):
  pool   = wk11_folds.json     targets.<seq>.threshold_only  (all folds pooled, fold-level bootstrap, ci_unit fold)
  folds  = ../../<private>/paper/phase4/calibration_<seq>.json  per-fold fixed_cell_f1, oracle_f1/oracle_cell and
           the per-draw test_f1 of the CAL / CAL-NEG variants (draws are averaged per fold here)
Both were produced by ../../<private>/paper/phase4/calibrate_n0.py (predict_n0.py supplies the raw VALDO predictions);
the fold definition is phase4/chains_frozenA1v2.json, frozen, unchanged.

Output: wk11_thresholds.json
    python wk11_02_thresholds.py
"""
from __future__ import annotations
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = _P + "/paper/phase4"
NS = ["3", "5", "10", "15", "20", "30", "40", "all"]


def load_pooled():
    d = json.load(open(f"{HERE}/wk11_folds.json"))
    return d


def per_fold(mod):
    """Per fold: the n = 0 point, the test-scan upper bound and every CAL / CAL-NEG draw's test F1."""
    d = json.load(open(f"{P4}/calibration_{mod}.json"))
    out = {}
    for k, fk in d["folds"].items():
        draws = fk["draws"]
        agg = {}
        for key, v in draws.items():
            if key.startswith("n"):
                fam, label = "cmb_free", key.split("_")[0][1:]
            elif key.startswith("neg_"):
                fam, label = "annotated", key.split("_")[1]
            else:
                continue
            agg.setdefault((fam, label), []).append(v["test_f1"])
        out[k] = dict(test_scans=fk["test_scans"], fixed_cell_f1=fk["fixed_cell_f1"], oracle_f1=fk["oracle_f1"],
                      oracle_cell=fk["oracle_cell"],
                      cmb_free={l: sum(a) / len(a) for (f, l), a in agg.items() if f == "cmb_free"},
                      annotated={l: sum(a) / len(a) for (f, l), a in agg.items() if f == "annotated"})
    return out


def api(t):
    """The fields of one pooled threshold_only row that the manuscript cites."""
    return dict(f1=t["f1"], f1_ci95=t.get("f1_ci95"), fp_per_scan=t["fp_per_scan"],
                fp_per_scan_ci95=t.get("fp_per_scan_ci95"), n_folds=t.get("n_folds"),
                n_scans_total=t.get("n_scans_total"), ci_unit=t.get("ci_unit"))


def pooled_variant(pool, mod, name):
    src = pool["targets"][mod]["threshold_only"][name]
    return {n: api(v) for n, v in src.items()
            if isinstance(v, dict) and (n[0].isdigit() or n == "all")}


def weighted(rows, key, wkey="test_scans"):
    tot = sum(r[wkey] for r in rows)
    return sum(r[key] * r[wkey] for r in rows) / tot


def per_fold_check(pf, mod, pool):
    """Does CMB-free calibration beat n = 0 in EVERY fold, not only on the pooled average?"""
    res = {}
    n0 = pool["targets"][mod]["threshold_only"]["N0"]["0"]["f1"]
    for label in NS:
        deltas, wins = [], 0
        for k, fk in sorted(pf.items()):
            got = fk["cmb_free"].get(label)
            base = fk["fixed_cell_f1"]
            if got is None:
                continue
            deltas.append(round(got - base, 4))
            wins += int(got > base)
        res[label] = dict(n_folds=len(deltas), folds_above_n0=wins,
                          min_delta=min(deltas) if deltas else None, max_delta=max(deltas) if deltas else None,
                          deltas_by_fold=deltas)
    n0_fold = [round(fk["fixed_cell_f1"], 4) for _, fk in sorted(pf.items())]
    return dict(pooled_n0_f1=n0, n0_f1_by_fold=n0_fold, per_n=res)


def main():
    pool = load_pooled()
    out = dict(
        generated="2026-11-10", step="wk11b step 2 of 6 -- threshold recalibration (one set of numbers)",
        source=dict(
            pooled="paper/wk11_folds.json targets.<seq>.threshold_only (all folds pooled, fold-level bootstrap, ci_unit fold)",
            per_fold="<private>/paper/phase4/calibration_<seq>.json (aggregate keys only; per-fold fixed_cell_f1, oracle_f1, draw test_f1)",
            code="<private>/paper/phase4/calibrate_n0.py + predict_n0.py (unchanged); folds chain_frozenA1v2.json, frozen"),
        definition=dict(
            threshold_only="the network is unchanged (n = 0 weights, VALDO full data); only threshold and minimum size are re-chosen, in each fold on that fold's training scans",
            cmb_free="(a) threshold re-chosen on n CMB-free training scans -- no annotation needed",
            annotated="(b) threshold re-chosen on n affected (annotated) training scans",
            test_scan_upper_bound="(c) best cell chosen on the fold's TEST scans themselves -- not attainable in practice, upper bound only",
            fp_per_scan="false CMB candidates per scan after post-processing (threshold + minimum size)",
            note="the n = 0 point uses the fixed cell 0.3 / 2 mm3 as downloaded"),
        n0={}, cmb_free={}, annotated={}, test_scan_upper_bound={}, per_fold_check={}, statements={})
    for mod in ["t2star", "swi"]:
        pf = per_fold(mod)
        cf, an = pooled_variant(pool, mod, "CAL"), pooled_variant(pool, mod, "CAL-NEG")
        out["n0"][mod] = api(pool["targets"][mod]["threshold_only"]["N0"]["0"])
        out["cmb_free"][mod] = cf
        out["annotated"][mod] = an
        rows = [dict(test_scans=fk["test_scans"], oracle_f1=fk["oracle_f1"], fixed_cell_f1=fk["fixed_cell_f1"])
                for _, fk in sorted(pf.items())]
        out["test_scan_upper_bound"][mod] = dict(
            pooled_oracle_f1=round(weighted(rows, "oracle_f1"), 4),
            pooled_fixed_cell_f1=round(weighted(rows, "fixed_cell_f1"), 4),
            oracle_f1_by_fold=[round(fk["oracle_f1"], 4) for _, fk in sorted(pf.items())],
            oracle_cell_by_fold=[fk["oracle_cell"] for _, fk in sorted(pf.items())],
            by_fold={k: dict(test_scans=fk["test_scans"], oracle_f1=round(fk["oracle_f1"], 4),
                             fixed_cell_f1=round(fk["fixed_cell_f1"], 4), oracle_cell=fk["oracle_cell"])
                     for k, fk in sorted(pf.items())},
            how="scan-weighted mean over the five folds; the per-fold value is the best cell of that fold's test scans")
        out["per_fold_check"][mod] = per_fold_check(pf, mod, pool)
        # one short sentence per sequence, the same words go into 3.2, 3.3 and 4.2
        c = out["cmb_free"][mod]
        out["statements"][mod] = dict(
            n0=f"n = 0: F1 {out['n0'][mod]['f1']:.2f}, {out['n0'][mod]['fp_per_scan']:.1f} false CMB per scan",
            cmb_free_n3_15=f"CMB-free threshold (a): F1 {min(c[n]['f1'] for n in ['3','5','10','15']):.2f}-"
                           f"{max(c[n]['f1'] for n in ['3','5','10','15']):.2f}, false CMB per scan "
                           f"{min(c[n]['fp_per_scan'] for n in ['3','5','10','15']):.1f}-"
                           f"{max(c[n]['fp_per_scan'] for n in ['3','5','10','15']):.1f} for n = 3 to 15",
            cmb_free_helps_every_fold=all(v["folds_above_n0"] == v["n_folds"] for v in out["per_fold_check"][mod]["per_n"].values()),
            upper_bound=f"test-scan upper bound (c): F1 {out['test_scan_upper_bound'][mod]['pooled_oracle_f1']:.2f} "
                        f"(as downloaded with the fixed cell {out['test_scan_upper_bound'][mod]['pooled_fixed_cell_f1']:.2f})")
    json.dump(out, open(f"{HERE}/wk11_thresholds.json", "w"), indent=1)
    # short summary, aggregates only
    for mod in ["t2star", "swi"]:
        print(f"### {mod}: {out['statements'][mod]['n0']}  |  {out['statements'][mod]['upper_bound']}")
        def _ci(v):
            c = v.get("f1_ci95")
            return f"[{c[0]:.2f},{c[1]:.2f}]" if c else "[no ci]"
        print(f"    CMB-free  (a): " + " ".join(
            f"n{n}: F1 {out['cmb_free'][mod][n]['f1']:.2f} {_ci(out['cmb_free'][mod][n])} "
            f"{out['cmb_free'][mod][n]['fp_per_scan']:.1f}FP" for n in ["3", "5", "10", "15", "20", "30", "40", "all"]))
        print(f"    annotated (b): " + " ".join(
            f"n{n}: F1 {out['annotated'][mod][n]['f1']:.2f} {out['annotated'][mod][n]['fp_per_scan']:.1f}FP"
            for n in ["3", "5", "10", "15"] if n in out['annotated'][mod]))
        print(f"    helps in every fold: {out['statements'][mod]['cmb_free_helps_every_fold']}  " + " ".join(
            f"n{n}:{out['per_fold_check'][mod]['per_n'][n]['folds_above_n0']}/{out['per_fold_check'][mod]['per_n'][n]['n_folds']}"
            for n in ["3", "5", "10", "15", "20"]))
        print(f"    oracle by fold: {out['test_scan_upper_bound'][mod]['oracle_f1_by_fold']}")
    print("written", f"{HERE}/wk11_thresholds.json")


if __name__ == "__main__":
    main()
