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
# wk11_02_margins.py -- step 1b of the wk11 plan (10.11.2026).
# WHAT: make the DECISION THRESHOLDS of the two pre-specified rules explicit and citable:
#       the plateau margin, the three criteria of the 2.6 rule (with their margins) and, for each sequence,
#       the value that decided n -- so "criteria met at 10" can be checked against a named threshold.
# WHERE: input  MB/paper/phase4/primary_endpoint.json  (plateau rule, per-n lower bound)
#                 MB/paper/phase4/good_for_us.json       (rule of 2.6, per-n values)
#        output  wk11_margins.json
# HOW:   reads only the rule strings and the per-n scalars; prints a short summary, never per-case data.
import json, os

P4 = _P + "/paper/phase4"
OUT = _R + "/paper/wk11_margins.json"

prim = json.load(open(os.path.join(P4, "primary_endpoint.json")))
gfus = json.load(open(os.path.join(P4, "good_for_us.json")))

out = {
    "source": {"plateau": "phase4/primary_endpoint.json", "good_for_us": "phase4/good_for_us.json"},
    "plateau_rule": {
        "margin": -0.02,
        "text": ("smallest n whose paired case-bootstrap one-sided 95 % lower bound of F1(n) - F1(all) "
                 "is above -0.02, holding at n and at every larger n"),
        "reference": "F1(all scans) of the same variant",
    },
    "good_for_us_rule": {
        "c1": "F1 >= 95 % of the reference chain F1 of the study's own model (wk2 S/P control, in-site, all scans)",
        "c1_factor": 0.95,
        "c2": "one-sided 95 % lower bound (case bootstrap, 2000) of F1 - best external baseline (SHIVA-CMB v2 nested) > -0.02",
        "c2_margin": -0.02,
        "c3": "ICC(3,1) of the subject-level counts >= 0.75",
        "c3_icc_min": 0.75,
        "text": gfus.get("criteria"),
    },
    "reference_chain_f1": gfus.get("reference_f1"),
    "external_best": gfus.get("ext_best"),
    "per_sequence": {},
}

for seq in ["t2star", "swi"]:
    ch = prim["targets"][seq]["chain"].get("FT", {})
    gf = gfus["targets"][seq].get("FT", {})
    rows = {}
    for n, v in gf.get("per_n", {}).items():
        p = ch.get("per_n", {}).get(n, {})
        rows[n] = {
            "f1": round(v["f1"], 4),
            "c1_ok": v.get("c1"), "c2_ok": v.get("c2"), "c3_ok": v.get("c3"),
            "lb95_F1_minus_ext": (round(v["lb95_F1_minus_ext"], 4)
                                  if v.get("lb95_F1_minus_ext") is not None else None),
            "icc31": v.get("icc31"),
            "lb95_F1n_minus_F1all": (round(p["lb95_F1n_minus_F1all"], 4)
                                     if p.get("lb95_F1n_minus_F1all") is not None else None),
        }
    out["per_sequence"][seq] = {
        "variant": "FT (fine-tuned from the public model)",
        "plateau_n": ch.get("plateau_n"),
        "good_for_us_n": gf.get("good_for_us_n"),
        "per_n": rows,
    }

json.dump(out, open(OUT, "w"), indent=1, sort_keys=False)

print("wrote", OUT)
for seq in ["t2star", "swi"]:
    s = out["per_sequence"][seq]
    print("  %-7s plateau_n=%s good_for_us_n=%s" % (seq, s["plateau_n"], s["good_for_us_n"]))
    for n in ["10", "40"]:
        r = s["per_n"].get(n)
        if r:
            print("    n=%-3s F1 %.3f  lb(F1-ext) %+.3f (margin -0.02)  lb(F1(n)-F1(all)) %+.3f  ICC %.4f  c=%s%s%s"
                  % (n, r["f1"], r["lb95_F1_minus_ext"], r["lb95_F1n_minus_F1all"], r["icc31"],
                     "1" if r["c1_ok"] else ".", "2" if r["c2_ok"] else ".", "3" if r["c3_ok"] else "."))
