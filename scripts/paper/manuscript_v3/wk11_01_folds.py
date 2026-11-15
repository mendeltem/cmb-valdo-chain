#!/usr/bin/env python
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
# wk11_01_folds.py -- step 1 of the wk11 plan (09.11.2026).
# WHAT: F1 and false positives per scan at n = 0, 3, 5, 10, 15, 20, 30, 40 for both sequences and all variants,
#       pooled over ALL FIVE frozen test folds and ALL draws, with case-level bootstrap intervals.
#       No new training, no change to any result file: reads result files already on disk.
# WHERE: input  MB/paper/phase4/chains_frozenA1v2.json   (frozen-A1v2 = frozen-V(a) stage-2 chain)
#                 MB/paper/phase4/calibration_{seq}_blocks.json  (variant N0 = as downloaded, n = 0;
#                 CAL = threshold re-chosen on CMB-free scans, CAL-NEG = threshold re-chosen on affected scans)
#                 MB/paper/phase4/primary_endpoint.json   (plateau n, n = all)
#                 MB/paper/phase4/good_for_us.json         ("good enough for practice" n, rule of 2.6)
#                 MB/paper/phase4_eval_{t2s,swi}_n0.json   (n = 0 reference: VALDO full-data model)
#        output  wk11_folds.json
# HOW:   per scan the JSON gives (tp, fp, fn). Counts are pooled over folds (folds partition the scans) and
#        averaged over the draws/seeds of the same n (a draw = one trained model; every n sees the same test scans).
#        Intervals: case-level bootstrap of 2000 resamples of the pooled scan list, seed 7 -> reproducible.
#        For n = 0 and the threshold-only variants only fold-level sums exist, so their interval is a
#        fold-level bootstrap (5 folds) and is marked as such.
# NOTE: Own-cohort aggregates only, no case ids; the manuscript cites only the aggregates in wk11_folds.json.
import json, os, random

P4 = _P + "/paper/phase4"
PM = _P + "/paper"
OUT = _R + "/paper/wk11_folds.json"
NS = [0, 3, 5, 10, 15, 20, 30, 40]        # training-set sizes with a chain run
NALL = {"t2star": "60", "swi": "49"}      # n = all scans (from primary_endpoint.json n_all)
CH = ["S", "FT", "FTn"]                   # S = from scratch, FT = fine-tune, FTn = fine-tune normalisation only
DRAW_SUFFIX = ("HIGH", "LOW", "MID", "NONE")   # FT draw strategies; _NONE is also a real draw of FT


def f1_of(tp, fp, fn):
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def boot(cases, B=2000, seed=7):
    """case-level bootstrap over the pooled scan list; each entry is [tp, fp, fn]."""
    rnd = random.Random(seed)
    n = len(cases)
    if n < 2:
        return None, None
    f1s, fps = [], []
    for _ in range(B):
        tp = fp = fn = 0
        for _ in range(n):
            c = cases[rnd.randrange(n)]
            tp += c[0]; fp += c[1]; fn += c[2]
        f1s.append(f1_of(tp, fp, fn)); fps.append(fp / n)
    f1s.sort(); fps.sort()
    return ([round(f1s[int(0.025 * B)], 4), round(f1s[int(0.975 * B)], 4)],
            [round(fps[int(0.025 * B)], 4), round(fps[int(0.975 * B)], 4)])


def agg(cases, unit="case", n_scans=None):
    tp = sum(c[0] for c in cases); fp = sum(c[1] for c in cases); fn = sum(c[2] for c in cases)
    f1_ci, fp_ci = boot(cases)
    n = len(cases)
    den = n_scans if n_scans else n          # per-scan denominator: scans, not bootstrap units (folds)
    return {"n_scans": n, "tp": round(tp, 2), "fp": round(fp, 2), "fn": round(fn, 2),
            "f1": round(f1_of(tp, fp, fn), 4),
            "fp_per_scan": round(fp / den, 4) if den else None,
            "f1_ci95": f1_ci, "fp_per_scan_ci95": fp_ci, "ci_unit": unit}


def parse_chain_key(key):
    """'t2star/FT_n10_fold0_s1_HIGH' -> (seq, var, nstr, fold, draw) ; None if not a chain key."""
    if "/" not in key:
        return None
    seq, rest = key.split("/", 1)
    parts = rest.split("_")
    if len(parts) < 4 or not parts[1].startswith("n"):
        return None
    var, nstr = parts[0], parts[1][1:]
    if not parts[2].startswith("fold") or not parts[3].startswith("s"):
        return None
    fold = int(parts[2][4:])
    draw = parts[4] if len(parts) > 4 else ""
    return seq, var, nstr, fold, draw


def pool_chain(chains, seq, var, nstr):
    """Pool one (seq, var, n) over all folds and draws. Returns per-scan cases averaged over the draws."""
    per_scan = {}
    for k, v in chains.items():
        p = parse_chain_key(k)
        if not p or p[0] != seq or p[1] != var or p[2] != nstr:
            continue
        if p[4] in DRAW_SUFFIX:   # fix 11.11.2026: headline curve = default stratified draw only (draw_strategies.json note);
            continue              # the HIGH/LOW/MID/NONE arms were averaged in before, lowering FT at n = 5, 10, 20
        pc = v.get("a_pooled", {}).get("per_case", {})
        if not pc:
            continue
        fold = p[3]
        for case, vals in pc.items():
            acc = per_scan.setdefault((fold, case), [0.0, 0.0, 0.0, 0])
            acc[0] += vals[0]; acc[1] += vals[1]; acc[2] += vals[2]; acc[3] += 1
    return [[a[0] / a[3], a[1] / a[3], a[2] / a[3]] for a in per_scan.values()]


def pool_blocks(blocks, variant, nstr=None):
    """Fold-level sums for a threshold-only variant at ONE n. Same scheme as pool_chain: folds partition the
    scans, so tp/fp/fn are summed over folds for each seed and averaged over the seeds (draws).
    Bootstrap unit = fold. Returns (per_fold_cases, total_scans)."""
    rows = [v for v in blocks.values() if v.get("variant") == variant
            and (nstr is None or str(v.get("n")) == nstr)]
    if not rows:
        return [], 0
    seeds = sorted(set(v.get("seed") for v in rows))
    folds = sorted(set(v.get("fold") for v in rows))
    per_fold = {}
    for f in folds:
        acc = [0.0, 0.0, 0.0]
        k = 0
        for s in seeds:
            hit = [v for v in rows if v.get("fold") == f and v.get("seed") == s]
            if not hit:
                continue
            acc[0] += sum(v["tp"] for v in hit); acc[1] += sum(v["fp"] for v in hit)
            acc[2] += sum(v["fn"] for v in hit); k += 1
        if k:
            per_fold[f] = [acc[0] / k, acc[1] / k, acc[2] / k]
    # scans of the test fold: same for every seed; sum over folds (folds partition the scans)
    scans = sum(int(v["scans"]) for v in rows if v.get("seed") == seeds[0])
    return [per_fold[f] for f in folds if f in per_fold], scans


chains = json.load(open(os.path.join(P4, "chains_frozenA1v2.json")))
prim = json.load(open(os.path.join(P4, "primary_endpoint.json")))
gfus = json.load(open(os.path.join(P4, "good_for_us.json")))

out = {
    "source": {
        "chains": "phase4/chains_frozenA1v2.json",
        "n0_and_threshold_only": "phase4/calibration_{seq}_blocks.json",
        "plateau_and_n_all": "phase4/primary_endpoint.json",
        "good_for_us": "phase4/good_for_us.json",
        "n0_reference": "phase4_eval_{t2s,swi}_n0.json",
        "n_all": NALL,
        "intervals": "case-level bootstrap, 2000 resamples, seed 7; folds pooled; draws averaged per scan. "
                     "n = 0 and the threshold-only variants carry only fold-level sums -> fold-level bootstrap "
                     "(5 folds), field ci_unit = 'fold'. Threshold-only variants are stored PER n (the threshold "
                     "is re-chosen within each training-set size). FTn has no n = all by design (only n = 5, 10, 20) "
                     "and therefore no plateau n.",
        "variants": {"S": "trained from scratch", "FT": "fine-tuned", "FTn": "fine-tuned, normalisation only",
                     "N0": "pretrained model as downloaded (n = 0)",
                     "CAL": "threshold re-chosen on CMB-free scans",
                     "CAL-NEG": "threshold re-chosen on affected scans"},
    },
    "targets": {},
}

for seq in ["t2star", "swi"]:
    tgt = out["targets"].setdefault(seq, {})
    # --- chain variants over training-set size n ---
    for var in CH:
        per_n = {}
        for n in NS:
            if n == 0:
                continue
            cases = pool_chain(chains, seq, var, str(n))
            if cases:
                per_n[str(n)] = agg(cases)
        cases = pool_chain(chains, seq, var, "all")
        if cases:
            per_n["all"] = agg(cases)
        if per_n:
            per_n["_n_all_expected"] = NALL[seq]
            tgt[var] = per_n
    # --- n = 0 and the threshold-only variants, PER n (a threshold re-chosen is a separate condition) ---
    blocks = json.load(open(os.path.join(P4, "calibration_%s_blocks.json" % seq)))
    thr = {}
    for var, label in [("N0", "pretrained model as downloaded (n = 0)"),
                       ("CAL", "threshold re-chosen on CMB-free scans"),
                       ("CAL-NEG", "threshold re-chosen on affected scans")]:
        ns = sorted(set(str(v.get("n")) for v in blocks.values() if v.get("variant") == var),
                    key=lambda s: (s == "all", int(s) if s.isdigit() else 0))
        per_n = {}
        for nstr in ns:
            cases, scans = pool_blocks(blocks, var, nstr)
            if not cases:
                continue
            a = agg(cases, unit="fold", n_scans=scans)
            a["what"] = label
            a["n_scans_total"] = scans
            a["n_folds"] = len(cases)
            seeds = sorted(set(v.get("seed") for v in blocks.values()
                               if v.get("variant") == var and str(v.get("n")) == nstr))
            a["seeds"] = seeds
            per_n[nstr] = a
        if per_n:
            thr[var] = per_n
    if thr:
        tgt["threshold_only"] = thr
        if "CAL-NEG" in thr and any(n in thr["CAL-NEG"] for n in ("20", "21", "22")):
            thr["CAL-NEG"]["_note"] = ("the last CAL-NEG run is stored under two labels on disk "
                                       "(n = 20/21 for t2star, 20/21/22 for swi); the rows are reported as found")

# --- plateau n (pre-specified rule) and good_for_us n (rule of 2.6) per sequence and variant ---
for seq in ["t2star", "swi"]:
    tgt = out["targets"][seq]
    chain = prim["targets"][seq]["chain"]
    gf = gfus["targets"][seq]
    for var in CH:
        if var in tgt:
            pd = chain.get(var, {})
            tgt[var]["plateau_n"] = pd.get("plateau_n")
            tgt[var]["plateau_note"] = pd.get("status") or (
                "smallest n whose paired case-bootstrap one-sided 95 % lower bound of F1(n) - F1(all) > -0.02, "
                "at n and every larger n")
            tgt[var]["f1_all"] = pd.get("f1_all")
            tgt[var]["fit"] = pd.get("fit")
            tgt[var]["good_for_us_n"] = gf.get(var, {}).get("good_for_us_n")
    out["plateau_rule"] = ("smallest n whose paired case-bootstrap one-sided 95 % lower bound of "
                           "F1(n) - F1(all) > -0.02, holding at n and every larger n")
    out["good_for_us_rule"] = gfus.get("criteria")
    out["reference_chain_f1"] = gfus.get("reference_f1")
    out["external_best"] = gfus.get("ext_best")

# --- n = 0 reference: VALDO full-data model on the in-house cohort ---
n0ref = {}
for seq, fn in [("t2star", "phase4_eval_t2s_n0.json"), ("swi", "phase4_eval_swi_n0.json")]:
    d = json.load(open(os.path.join(PM, fn)))
    runs = {}
    for run, models in d.get("runs", {}).items():
        for model, m in models.items():
            runs[run] = {"cases": m["cases"], "f1": m["f1"], "precision": m["precision"],
                         "sensitivity": m["sensitivity"], "fp_per_case": m["fp_per_case"]}
    n0ref[seq] = {"file": fn, "post_processing": d.get("post_processing"), "runs": runs}
out["n0_reference_valdo_full_data"] = n0ref

json.dump(out, open(OUT, "w"), indent=1)

# short, aggregate-only summary on stdout
for seq, d in out["targets"].items():
    print("==", seq)
    for var in CH:
        per_n = d.get(var)
        if not per_n:
            continue
        for n in ["3", "5", "10", "15", "20", "30", "40", "all"]:
            if n in per_n:
                a = per_n[n]
                ci = ("%.3f-%.3f" % (a["f1_ci95"][0], a["f1_ci95"][1])) if a["f1_ci95"] else "n/a"
                print("  %-4s n=%-4s F1 %.3f [%s] fp/scan %.2f scans %d" %
                      (var, n, a["f1"], ci, a["fp_per_scan"], a["n_scans"]))
        print("  %-4s plateau_n=%s good_for_us_n=%s" % (var, per_n.get("plateau_n"), per_n.get("good_for_us_n")))
    for var, per_n in d.get("threshold_only", {}).items():
        print("  -- %s" % var)
        for nstr, a in per_n.items():
            if nstr.startswith("_"):
                continue
            print("     n=%-4s F1 %.3f fp/scan %.2f (fold-level CI, %d folds, seeds %s) scans %d" %
                  (nstr, a["f1"], a["fp_per_scan"], a["n_folds"], a["seeds"], a["n_scans_total"]))
print("written", OUT)
