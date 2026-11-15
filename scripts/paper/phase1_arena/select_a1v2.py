"""wk9 STEP 1: selection.json for Arena 1 v2 from the two pooled VALDO-only runs (seeds 0 and 1).

Applies cmb.stage2.tset_summary (rules v2, control 'pooled tset-plain') to the arm runs,
then the pre-registered winner rule (ARBEIT.md '## wk9', written before any score was seen):
  winner = the arm with decision 'carry to confirmation' AND the largest shrunk delta;
  if none carries -> the CONTROL stays (tset-plain = the paper chain, a valid result);
  if two are within 0.005 shrunk delta -> the one with fewer inputs.
Also writes the frozen threshold rule, the provenance fields (training_data, licence) and
the design-leak-test bookkeeping. No in-house input anywhere.

    python select_a1v2.py
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
import glob
import json
import os
import sys

ROOT = _R
MB = _P
sys.path.insert(0, ROOT)
from cmb.stage2.tset_summary import load, summarise          # noqa: E402

OUT = f"{MB}/stage2/a1v2/selection.json"

# pre-registered input counts per arm (fewer inputs = simpler, per the tie rule)
INPUTS = {
    "pooled tset-plain": "image patches (stage-1 candidate crop), no tabular features",
    "pooled feat-vessel": "tabular features + per-scan neighbour features (no patches)",
    "pooled feat-vessel-tissue": "tabular features + neighbour features + 5 tissue one-hot",
}
INPUT_N = {"pooled tset-plain": 3, "pooled feat-vessel": 11, "pooled feat-vessel-tissue": 16}
ARM_SHORT = {m: m.replace("pooled ", "") for m in INPUTS}


def main():
    files = sorted(glob.glob(f"{MB}/stage2/a1v2/tset_a1v2_all_s[01].json"))
    assert len(files) == 2, f"expected the seed 0 and 1 runs, found {files}"
    runs = load(files)
    control = "pooled tset-plain"
    assert control in runs or (0, control) in runs
    summ = summarise(runs, control)
    # the two probability baselines of the same runs, for context (not part of the rule)
    baselines = {}
    for f in files:
        d = json.load(open(f))
        for base in ("probability", "probability+csf"):
            if base in d["per_case"]:
                baselines[base] = d["per_case"][base]

    carry = [m for m, e in summ.items() if e["decision"] == "carry to confirmation"]
    if carry:
        best = max(carry, key=lambda m: summ[m]["shrunk_delta"])
        rivals = [m for m in carry if abs(summ[m]["shrunk_delta"] - summ[best]["shrunk_delta"]) < 0.005]
        if len(rivals) > 1:
            best = min(rivals, key=lambda m: INPUT_N[m])
            reason = (f"{best}: 'carry to confirmation'; shrunk deltas of {rivals} within 0.005 -> "
                      f"the one with fewer inputs ({INPUT_N[best]}) wins")
        else:
            reason = f"{best}: 'carry to confirmation' and the largest shrunk delta " \
                     f"({summ[best]['shrunk_delta']} > 0) among {carry}"
    else:
        best = control
        reason = ("no arm with decision 'carry to confirmation' -> the CONTROL stays "
                  "(= the paper chain, a valid result), per the pre-registered rule")
    winner = ARM_SHORT[best]

    # frozen classifier: pooled VALDO OOF threshold, variant (a), seed offset 0 (as frozen-V)
    sel = {
        "winner": winner,
        "reason": reason,
        "rule": "cmb.stage2.tset_summary rules v2 (control 'pooled tset-plain'); winner = 'carry to confirmation' "
                "and largest shrunk delta; none carries -> control stays; within 0.005 shrunk delta -> fewer inputs. "
                "Pre-registered in ARBEIT.md '## wk9' before any score was seen.",
        "arms_run": sorted(summ) + [control],
        "arms_dropped_before_runs": {
            "feat-vessel-atlas": "no venous-atlas values exist for the Phase-2 target blocks (0/75 T2*, 0/61 SWI registered)",
            "feat-vessel-tissue-atlas": "as above",
            "feat-ens-tissue-atlas": "as above (needs the tissue-atlas feature basis)",
            "feat-mine-tissue-atlas": "as above (R04 basis = tissue + 3 venous-atlas values)"},
        "summary": summ,
        "control": control,
        "probability_baselines_valdo": {b: bases["valdo"] for b, bases in baselines.items()},
        "frozen_classifier": {
            "arm": winner,
            "training": "VALDO pool only (ergebnisse/stage2/a03-aniso-e60-gd-weich-z40), seeds 0 and 1 for selection, "
                        "seed offset 0 for the frozen weights; nested threshold as always",
            "threshold_rule": "pooled VALDO out-of-fold F1 threshold, applied unchanged (variant (a), as frozen-V); "
                              "variant (b) = per-cohort threshold of the same pool OOF scores",
        },
        "design_leak_test": {
            "statement": "design list informed by Arena 2",
            "outcome": (f"VALDO CV picked '{winner}'"
                        + (" (an Arena-2 feature arm: the in-house knowledge did not change the decision)"
                           if winner != "tset-plain" else " (the pre-Arena-2 control: the control is the winner)")),
        },
        "training_data": {
            "name": "VALDO Task 2, manifest dev/manifest_valdo_gitter_d.json (57 of the 72 VALDO Task-2 cases; "
                    "the 15 held-out cases of 28.09.2026 are not used)",
            "count": 57,
            "lesions": 139,
            "no_hospital_data": True,
        },
        "licence": "weights trained on VALDO (CC BY-NC-SA 4.0); cite Sudre et al. (VALDO) and add 'for the ALFA Study'",
        "runs": files,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(sel, open(OUT, "w"), indent=1)
    print(f"winner: {winner}")
    print("reason:", reason)
    for m, e in sorted(summ.items()):
        print(f"  {m:28s} delta {e['delta_all']:+.3f} shrunk {e['shrunk_delta']:+.3f} "
              f"lower {e['simultaneous_lower']:+.3f} -> {e['decision']}")
    print("written to", OUT)


if __name__ == "__main__":
    main()
