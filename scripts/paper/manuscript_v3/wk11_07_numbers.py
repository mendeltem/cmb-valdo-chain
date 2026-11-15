#!/usr/bin/env python
"""wk11_07_numbers.py -- wk11f step 6: derived numbers for NUMBERS_v3.md.

CPU only, reads result files already on disk, writes NO result file.
Writes paper/wk11_numbers.json: one key path per number that the manuscript cites
but that is not stored as a single key in one of the wk11_*/<private> JSONs.
Own cohort: aggregates only, no case ids.
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
import json, os, sys

MB = _P
P = _R + "/paper"
OUT = os.path.join(P, "wk11_numbers.json")

def load(path):
    with open(path) as f:
        return json.load(f)

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

out = {"source": {
    "folds": "paper/wk11_folds.json",
    "decisions": "<private>/paper/decisions/chainA_s0.jsonl + chainA_s1.jsonl (mode pooled tset-plain, accepted only)",
    "patient": "<private>/paper/patient_threshold_summary.json",
    "clinical": "<private>/paper/clinical_summary.json",
    "n0_eval": "<private>/paper/phase4_eval_{t2s,swi}_n0.json",
    "extern": "<private>/extern/v2_{shiva,mednet,mbnet}_{swi,t2star}.json",
    "note": "aggregates only; the decision dumps are streamed, nothing per case is written here",
}}

# ---- sensitivity at the n = 0 operating point (as downloaded) ----
ev = {s: load(os.path.join(MB, "paper", "phase4_eval_%s_n0.json" % s)) for s in ("t2s", "swi")}
out["n0_as_downloaded"] = {
    "t2star": {"sensitivity": ev["t2s"]["runs"]["valdo72-on-t2s"][_LABEL + "-t2star"]["sensitivity"],
               "precision": ev["t2s"]["runs"]["valdo72-on-t2s"][_LABEL + "-t2star"]["precision"],
               "f1": ev["t2s"]["runs"]["valdo72-on-t2s"][_LABEL + "-t2star"]["f1"],
               "fp_per_case": ev["t2s"]["runs"]["valdo72-on-t2s"][_LABEL + "-t2star"]["fp_per_case"]},
    "swi": {"sensitivity": 522.0 / (522.0 + 130.0), "precision": 522.0 / (522.0 + 2262.0),
            "f1": 0.3038, "fp_per_case": 37.082},
}

# ---- external baselines as downloaded (author threshold and nested) ----
out["extern_baseline"] = {}
for name, tag in (("shiva", "SHIVA-CMB"), ("mednet", "MedNet-CMB"), ("mbnet", "MicrobleedNet")):
    out["extern_baseline"][name] = {}
    for seq, sfx in (("swi", "swi"), ("t2star", "t2star")):
        d = load(os.path.join(MB, "extern", "v2_%s_%s.json" % (name, sfx)))
        auth = d.get("components_author") or {}
        out["extern_baseline"][name][seq] = {
            "author_f1": auth.get("f1"), "author_sensitivity": auth.get("sensitivity"),
            "author_fp_per_case": auth.get("fp_per_case"),
            "nested_f1": d["nested"]["f1"], "nested_sensitivity": d["nested"]["sensitivity"],
            "nested_fp_per_case": d["nested"]["fp_per_case"],
            "nested_ci95": d["nested"].get("f1_ci95"),
        }

# ---- false positives: size, score, tissue and concentration per scan ----
fp = {}
for cohort in ("t2star", "swi"):
    false_vox, true_vox, false_score, true_score = [], [], [], []
    per_case_false = {}
    tissue_false, tissue_true = {}, {}
    for seed in (0, 1):
        path = os.path.join(MB, "paper", "decisions", "chainA_s%d.jsonl" % seed)
        with open(path) as fh:
            for line in fh:
                if '"cohort": "%s"' % cohort not in line:
                    continue
                r = json.loads(line)
                if r.get("mode") != "pooled tset-plain" or not r.get("accepted"):
                    continue
                case = r["case"]
                is_false = not r.get("touched")
                if is_false:
                    false_vox.append(r["voxels"])
                    false_score.append(r["score"])
                    per_case_false[case] = per_case_false.get(case, 0) + 1
                else:
                    true_vox.append(r["voxels"])
                    true_score.append(r["score"])
                    if case not in per_case_false:
                        per_case_false[case] = 0
    n_cases = len(per_case_false)
    n_zero = sum(1 for v in per_case_false.values() if v == 0)
    tot = sum(per_case_false.values())
    top = sorted(per_case_false.values(), reverse=True)
    k = max(1, int(round(0.2 * n_cases)))
    fp[cohort] = {
        "median_voxels_false": median(false_vox), "median_voxels_true": median(true_vox),
        "median_score_false": median(false_score), "median_score_true": median(true_score),
        "n_cases": n_cases, "n_cases_no_false": n_zero,
        "share_top20pct_cases": (sum(top[:k]) / tot) if tot else None,
        "share_false_small": (sum(1 for v in false_vox if v <= 40) / len(false_vox)) if false_vox else None,
        "tissue_false": tissue_false, "tissue_true": tissue_true,
    }
out["false_positives"] = fp

# ---- tissue of false / true accepted candidates (from the patient summary) ----
pts = load(os.path.join(MB, "paper", "patient_threshold_summary.json"))["false_positive_analysis"]["by_cohort"]
out["tissue_of_accepted_candidates"] = {}
for cohort in ("t2star", "swi"):
    t = pts[cohort]["tissue"]
    tot_t = sum(v["true"] for v in t.values())
    tot_f = sum(v["false"] for v in t.values())
    out["tissue_of_accepted_candidates"][cohort] = {
        "cortex_share_false": t["cortex"]["false"] / tot_f,
        "csf_share_false": t["CSF / ventricles"]["false"] / tot_f,
        "white_matter_share_false": t["cerebral white matter"]["false"] / tot_f,
        "deep_grey_matter_false": t["deep grey matter"]["false"],
        "deep_grey_matter_true": t["deep grey matter"]["true"],
        "deep_grey_matter_total": t["deep grey matter"]["false"] + t["deep grey matter"]["true"],
        "total_false": tot_f, "total_true": tot_t,
    }

# ---- patient level: threshold errors and the grey zone ----
ps = load(os.path.join(MB, "paper", "patient_threshold_summary.json"))
out["patient_level"] = {}
for cohort in ("t2star", "swi", "valdo"):
    lf = ps[cohort]["lesion_f1_operating_point"]
    out["patient_level"][cohort] = {
        "n": lf["n"], "tp": lf["tp"], "fn": lf["fn"], "tn": lf["tn"], "fp": lf["fp"],
        "sensitivity": lf["sensitivity"], "specificity": lf["specificity"],
        "sensitivity_ci": lf["sensitivity_ci"], "specificity_ci": lf["specificity_ci"],
        "n_above_threshold": lf["tp"] + lf["fp"],
        "fp_of_below": "%.0f of %.0f" % (lf["fp"], lf["tn"] + lf["fp"]),
    }
band = ps["radiologist_band_robustness"]
out["grey_zone"] = {"band": [3, 8], "nested_band_per_fold": band["nested_band_per_fold"]}

cl = load(os.path.join(MB, "paper", "clinical_summary.json"))["chains"]["A"]["by_dataset"]
out["clinical"] = {}
for cohort in ("t2star", "swi", "valdo"):
    c = cl[cohort]
    out["clinical"][cohort] = {
        "n": c["n"], "spearman_rho": c["count"]["spearman_rho"], "spearman_ci": c["count"]["spearman_ci"],
        "presence_sensitivity": c["presence"]["sensitivity"], "presence_specificity": c["presence"]["specificity"],
        "weighted_kappa_burden": c["burden_0_1to4_ge5"]["weighted_kappa"],
        "icc31": c["count"]["icc_31"],
    }

with open(OUT, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print("wrote", OUT)
print("false positives:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("tissue_false", "tissue_true")} for k, v in fp.items()}, indent=0))
