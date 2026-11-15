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
# wk11_04_prevalence.py -- step 4 of 6 of the wk11 plan (clinical CMB paper v3).
#
# Recomputes wk11_prevalence.json from the patient-level numbers that already exist on disk.
# CPU only, no local GPU, no result file is read per block and nothing is written outside $P.
#
# Input (aggregates only, never a case id):
#   MB/paper/phase4/a1v2_vs_frozenV.json  -> key "patient_level" only
#     (per modality: n, sensitivity, specificity, with_radiologist_band.{band, share_to_radiologist,
#      sensitivity, specificity})
#
# Decision under study (from patient_level.decision): more than 4 CMB excludes a patient from
# anti-amyloid antibody therapy; a pre-specified grey zone of chain counts 3-8 goes to a radiologist
# instead of being auto-excluded.
#
# What this is: plain arithmetic on measured aggregates. The prevalence p (the share of patients who
# truly have more than 4 CMB) is NOT known in advance for a new hospital -> p = 5 / 10 / 20 % are
# labelled ASSUMPTIONS, not measurements. The cohort is used only through the measured sensitivity
# and specificity at the operating point, and the measured share referred to the radiologist.
#
# Formulas (cohort sensitivity is 1.0, so the harmful error is a false exclusion):
#   TP   = p * sens
#   FP   = (1 - p) * (1 - spec)
#   PPV  = TP / (TP + FP)
#   share of eligible patients wrongly excluded = FP / (1 - p) = 1 - spec
#   share referred to the radiologist           = measured share_to_radiologist (constant, with band)
#
# Output: $P/wk11_prevalence.json (rewritten in place, same numbers as the 09.11.2026 version).

import json
import os

P = _R + "/paper"
SRC = _P + "/paper/phase4/a1v2_vs_frozenV.json"
SUMSRC = _P + "/paper/patient_threshold_summary.json"

PREVALENCES = [0.05, 0.10, 0.20]


def main():
    with open(SRC) as fh:
        pl = json.load(fh)["patient_level"]

    rows = {}
    for mod in ("swi", "t2star"):
        m = pl["per_modality"][mod]
        sens = float(m["sensitivity"])
        spec = float(m["specificity"])
        band = m["with_radiologist_band"]
        band_spec = float(band["specificity"])
        band_ref = round(float(band["share_to_radiologist"]), 3)

        by_p = {}
        for p in PREVALENCES:
            tp = p * sens
            fp = (1.0 - p) * (1.0 - spec)
            by_p["%.2f" % p] = {
                "ppv_excluded": round(tp / (tp + fp), 3),
                "wrongly_excluded_share": round(1.0 - spec, 3),
                "referred_to_radiologist_share": 0.0,
                "ppv_excluded_with_band": round((p * sens) / (p * sens + (1.0 - p) * (1.0 - band_spec)), 3),
                "wrongly_excluded_share_with_band": round(1.0 - band_spec, 3),
                "referred_to_radiologist_share_with_band": band_ref,
            }

        rows[mod] = {
            "n": int(m["n"]),
            "sensitivity": sens,
            "specificity": spec,
            "with_band": {
                "band": band["band"],
                "specificity": band_spec,
                "share_to_radiologist": band_ref,
            },
            "by_prevalence": by_p,
        }

    out = {
        "source_a1v2_vs_frozenV": SRC + " (patient_level.per_modality)",
        "source_counts": SUMSRC + " (lesion_f1_operating_point)",
        "decision": pl["decision"],
        "assumption": ("prevalence p = assumed share of patients who truly have > 4 CMB; "
                       "labelled as an assumption, plain arithmetic, no new runs"),
        "arithmetic": ("TP=p*sens; FP=(1-p)*(1-spec); PPV=TP/(TP+FP); "
                       "wrongly_excluded_share=FP/(1-p)=1-spec; "
                       "referred share = measured share_to_radiologist (constant, from the cohort)"),
        "band": "grey zone 3-8 -> referred to the radiologist instead of auto-excluded",
        "prevalences": PREVALENCES,
        "rows": rows,
    }

    dst = os.path.join(P, "wk11_prevalence.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
