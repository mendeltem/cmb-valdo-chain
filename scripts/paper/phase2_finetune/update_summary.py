"""Update MB/paper/phase4_summary.json from the computed endpoints (06.11.2026, R6 19:40):
reads phase4/primary_endpoint.json, good_for_us.json and draw_strategies.json, sets status done,
fills primary_endpoint (incl. the explicit stage-2-variant status), secondary_good_for_us (incl.
quantified scans-saved + SWI S-failure) and the draw-strategy arms; separates the consciously
skipped items (source diversity; secondary curve) so they are not lumped together. Keeps the
pre-registered text verbatim.
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

P4 = _P + "/paper/phase4"
S = _P + "/paper/phase4_summary.json"

d = json.load(open(S))
pe = json.load(open(f"{P4}/primary_endpoint.json"))
gf = json.load(open(f"{P4}/good_for_us.json"))
try:
    ds = json.load(open(f"{P4}/draw_strategies.json"))
except Exception:
    ds = None

d["status"] = ("INCOMPLETE -- reviewer point (R7 06.11.2026): the PRIMARY endpoint per "
               "'S/FT/FTn x frozen-V/updated' is delivered for the frozen-V column only; the "
               "updated/own-D column is NOT computed (HPC-GPU, deviation, no Freigabe). "
               "Everything else (frozen-V PRIMARY + SECONDARY, draw strategies, figures) is done; "
               "no new GPU runs this round.")
d["updated"] = "2026-11-06T20:05:00"
d["primary_endpoint"] = {
    "status": "INCOMPLETE (stage-2 variant frozen-V only; the 'updated' column of the headline table is NOT computed -- see stage2_variants; deviation + Freigabe needed)",
    "method": pe["method"],
    "n_values": ["0", "3", "5", "10", "15", "20", "30", "40", "all"],
    "variants_stage1": ["S", "FT", "FTn"],
    "variants_stage2": ["frozen-V (a: VALDO threshold)", "frozen-V (b: per-cohort threshold)"],
    "stage2_variants": {
        "frozen-V (a)/(b)": "done -- the delivered PRIMARY and SECONDARY curves (S/FT/FTn) are computed with stage-2 frozen-V(a); frozen-V(b) (per-cohort threshold) is stored per block in phase4/chains_frozenV.json.",
        "updated (JOINT on VALDO + the candidates of D's n training scans)": "NOT computed -- deviation, needs GPU (HPC): it needs the z40 candidate sets of the D TRAINING scans, which were never generated (run_block.py builds only the test-fold candidates; the training-fold U-Net predictions live in the 35.8 GB archived phase4/unet/). No number for 'updated' exists, and the headline table per variant (S/FT/FTn x frozen-V/updated) is therefore only half filled.",
        "own-D (classifier only on D, variant S)": "NOT computed -- same GPU dependency (D training-scan candidates).",
        "consequence": "The plateau-n result ('no variant reaches a plateau n < all') and every F1(n) curve in this file hold for stage-2 frozen-V(a) ONLY. With 'updated' the curves can differ (different classifier -> different F1(n) -> potentially a different plateau n); the PRIMARY definition ('not inferior to n = all of the SAME variant') must then be re-evaluated per variant."},
    "chain_f1": {m: {v: (pe["targets"][m]["chain"][v].get("curve") if v in pe["targets"][m]["chain"] else None)
                     for v in ("S", "FT", "FTn")} for m in ("t2star", "swi")},
    "plateau_n": {m: {v: pe["targets"][m]["chain"].get(v, {}).get("plateau_n")
                      for v in ("S", "FT", "FTn")} for m in ("t2star", "swi")},
    "fit_F1n_a_minus_b_n_pow_minus_c": {m: {v: pe["targets"][m]["chain"].get(v, {}).get("fit")
                                            for v in ("S", "FT", "FTn")} for m in ("t2star", "swi")},
    "per_n_lb95": {m: {v: pe["targets"][m]["chain"].get(v, {}).get("per_n")
                       for v in ("S", "FT", "FTn")} for m in ("t2star", "swi")},
    "note": ("No variant reaches a plateau n < all under the strict rule (paired case bootstrap LB95 of "
             "F1(n)-F1(all) > -0.02 at n and every larger n) -- for stage-2 frozen-V(a): t2star S n=30 LB -0.023, "
             "t2star FT n=40 LB -0.022, swi FT n=40 LB -0.015 (closest). Descriptive fit gives n at 95 % of the "
             "fitted plateau a: t2star S 33.8, t2star FT 10.9; for SWI the fit is nearly flat (small c), so n95(a) "
             "is not informative there. FTn has no n = all by design (spec: FTn only at n = 5, 10, 20) -> its "
             "plateau n is not defined. CAVEAT: all of this is stage-2 frozen-V(a) only (see stage2_variants)."),
    # Reviewer R7 (06.11.): the headline table must be a complete 3x2 grid per target -- every cell
    # carries either a number or an explicit machine-readable 'not computed' marker, so an automated
    # reader never hits a missing field.
    "headline_table": {
        "note": "Headline per target: plateau n (PRIMARY) and good-for-us n (SECONDARY) per variant "
                "S/FT/FTn x stage-2 frozen-V/updated. 'not_computed' = NOT computed (deviation, HPC-GPU, "
                "no Freigabe) -- deliberately a marker, not a missing field.",
        "t2star": {
            "S": {"plateau_n_frozenV": None, "good_for_us_n_frozenV": 15,
                  "plateau_n_updated": "not_computed", "good_for_us_n_updated": "not_computed"},
            "FT": {"plateau_n_frozenV": None, "good_for_us_n_frozenV": 10,
                   "plateau_n_updated": "not_computed", "good_for_us_n_updated": "not_computed"},
            "FTn": {"plateau_n_frozenV": None, "good_for_us_n_frozenV": 5,
                    "plateau_n_updated": "not_computed", "good_for_us_n_updated": "not_computed"},
        },
        "swi": {
            "S": {"plateau_n_frozenV": None, "good_for_us_n_frozenV": None,
                  "plateau_n_updated": "not_computed", "good_for_us_n_updated": "not_computed"},
            "FT": {"plateau_n_frozenV": None, "good_for_us_n_frozenV": 10,
                   "plateau_n_updated": "not_computed", "good_for_us_n_updated": "not_computed"},
            "FTn": {"plateau_n_frozenV": None, "good_for_us_n_frozenV": None,
                    "plateau_n_updated": "not_computed", "good_for_us_n_updated": "not_computed"},
        },
    },
    # Reviewer R7: the pre-registered expectation "~5-10 scans for T2* with FT" refers to FT in the
    # narrow sense (full fine-tuning), NOT to FTn (norm+output layers only). Judged per variant.
    "expectation_check_30_10": {
        "expectation": "T2* needs few scans (about 5-10) with FT; SWI needs clearly more, and the gain of FT over S is smaller (pre-registered 30.10.2026).",
        "t2star": {
            "FT": {"good_for_us_n": 10, "verdict": "within the band 5-10 (upper end; FT is the variant the expectation refers to)"},
            "FTn": {"good_for_us_n": 5, "verdict": "separate variant (norm+output layers only), NOT part of the 'FT' expectation; reported separately"},
            "S": {"good_for_us_n": 15, "verdict": "above the band, as expected (from scratch needs more)"},
        },
        "swi": {
            "FT": {"good_for_us_n": 10, "verdict": "consistent with 'SWI needs clearly more' (FT still 10, but only barely: criterion (2) margin -0.011 at n=10; the FT-over-S gain is small and S never qualifies at all)"},
            "FTn": {"good_for_us_n": None, "verdict": "never reaches criterion (1) (F1 ~0.60 < 0.632)"},
            "S": {"good_for_us_n": None, "verdict": "never reaches criterion (2) at any n (see scans_saved_by_pretraining.swi_detail)"},
        },
    },
}
d["secondary_good_for_us"] = {
    "status": "done (all three references present: phase2_summary.json, extern/summary.json, clinical_summary.json; stage-2 frozen-V(a))",
    "criteria": gf["criteria"],
    "reference_f1": gf["reference_f1"],
    "ext_best": gf["ext_best"],
    "good_for_us_n": {m: {v: gf["targets"][m][v]["good_for_us_n"] for v in ("S", "FT", "FTn")}
                      for m in ("t2star", "swi")},
    "per_n": {m: {v: gf["targets"][m][v]["per_n"] for v in ("S", "FT", "FTn")} for m in ("t2star", "swi")},
    "scans_saved_by_pretraining": {
        "definition": "headline = n of S minus n of FT at the SAME 'good for us' criterion",
        "t2star": 5,
        "t2star_detail": "S needs n=15, FT needs n=10 -> 15-10 = 5 scans saved",
        "swi": None,
        "swi_reason": "n/a - S erfuellt Kriterium (2) nie (LB95 von F1 - SHIVA-Baseline 0.637 bleibt bei jedem n unter -0.02, best n=all -0.027) -> n(S) existiert nicht, die Differenz S - FT ist keine Zahl; die einzige 'good for us'-Variante auf SWI ist FT (n=10)",
        "swi_detail": ("NOT DEFINED (and why): on SWI variant S never reaches the criterion at any measured n, "
                       "so n(S) does not exist and the difference is not a number. S fails criterion (2) alone "
                       "(F1 not below the best wk1 external baseline, LB95 > -0.02): the SHIVA-CMB v2 nested "
                       "baseline on SWI is F1 0.637, and S's chain LB95 of F1 - baseline stays below -0.02 at every "
                       "n: n=3 -0.188, n=5 -0.192, n=10 -0.084, n=15 -0.093, n=20 -0.062, n=30 -0.058, n=40 -0.040, "
                       "n=all(49) -0.027 (closest miss: 0.007 below the -0.02 line). Criteria (1) and (3) are met "
                       "from n=20 onward (F1 0.644-0.678 >= 0.632 = 95 % of reference 0.6649; ICC 0.995-0.997 >= 0.75). "
                       "FT reaches all three at n=10 (F1 0.688, LB95 -0.011, ICC 0.997). FTn never reaches criterion (1) "
                       "(F1 ~0.60 < 0.632 at n=5/10/20). Practical reading: on SWI the public (VALDO) model is the "
                       "only way to reach 'good for us' in the measured range; the number of scans saved by pretraining "
                       "is therefore 'S: not reached', not 0.")},
    "note": ("Headline: scans saved by starting from the public model: t2star 5 (S 15 - FT 10); SWI not defined "
             "(S never reaches criterion (2) at any n, best LB95 -0.027 at n=all; see scans_saved_by_pretraining).")}
d["draw_strategies"] = {
    "status": "done (computed R6 06.11. from the stored per-case counts; the 780 blocks DO include the strategy arms)",
    "note": ds["note"] if ds else "see phase4/draw_strategies.json",
    "targets": ds["targets"] if ds else None}
# consciously skipped items, kept SEPARATE (reviewer point: do not lump the secondary curve into source diversity)
d["consciously_not_run"] = {
    "source_diversity": {
        "what": "U-Net pre-trained on VALDO RSS only (34 scans, one scanner) vs all VALDO (72, three scanners), plus all-VALDO subsampled to 34; FT at n=10/20",
        "status": "NOT run",
        "reason": "SPEC: 'lowest priority, dropped first if the budget is short'; the measured ~190 GPU-h landed near the top of the estimate (190 +/-30 %, budget 220). The 780 blocks (core curve + draw strategies) were prioritised.",
        "question": "bewusst raus aus dem Paper oder nachrechnen (HPC-GPU; gleiche Draws, phase4_draws.py schreibt sie bereits)?"},
    "secondary_curve_own_candidates_fraction": {
        "what": "own candidates at fraction f vs + other datasets at 100 % (SPEC: 'Secondary, only if budget remains')",
        "status": "NOT run",
        "reason": "explicit SPEC condition 'only if budget remains' -- after the 780 U-Net blocks + stage-2 frozen-V chains no GPU budget remained in this job; it is a SEPARATE item from source diversity (own candidates of D at fraction f, not the U-Net pretraining source).",
        "question": "bewusst raus oder nachrechnen (HPC-GPU)?"},
    "stage2_updated_ownD": {
        "what": "stage-2 chains of the variants 'updated' (JOINT VALDO + D training-scan candidates) and 'own-D' (S only)",
        "status": "NOT run",
        "reason": "needs the D training-fold z40 candidate sets, which were never generated locally (run_block.py builds test-fold candidates only; training-fold U-Net predictions are in the 35.8 GB archived phase4/unet/) -> HPC-GPU.",
        "question": "nachrechnen (HPC) oder Freigabe, dass frozen-V(a) als Endpunkt des Papers genuegt?"}}
d["measured_points"] = {
    "note": "superseded by primary_endpoint.json / good_for_us.json (computed 06.11.2026 from the stored predictions)",
    "kept_for_provenance": True}

json.dump(d, open(S, "w"), indent=1)
print("updated", S)
