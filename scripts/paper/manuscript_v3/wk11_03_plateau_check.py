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
# wk11_03_plateau_check.py -- step 1c of the wk11 plan (10.11.2026, reviewer round).
# WHAT: make the numbers of the plateau sentence in 3.3/4.2/abstract citable and check them against the
#       source file. primary_endpoint.json stores plateau_n = null, but at n = all the lower bound is 0.0 by
#       construction (F1(n) - F1(all) with n = all), so "not reached" is only true for a plateau STRICTLY
#       below the full chain. This script prints, per sequence and variant, the per-n lower bound and which
#       reading the stored plateau_n supports.
# WHERE: input  MB/paper/phase4/primary_endpoint.json (per_n lower bounds, stored plateau_n, n_all)
#        output  wk11_plateau_check.json
# HOW:   reads only per-n scalars; prints a short table. No result file is touched, no case ids.
import json, os

P4 = _P + "/paper/phase4"
OUT = _R + "/paper/wk11_plateau_check.json"
N_ORDER = [3, 5, 10, 15, 20, 30, 40]
MARGIN = -0.02

prim = json.load(open(os.path.join(P4, "primary_endpoint.json")))
out = {"source": "phase4/primary_endpoint.json", "margin": MARGIN,
       "note": ("plateau_n is null for every variant. At n = all the lower bound is 0.0 by construction, so the "
                "strict rule as written ('at n and every larger n') would return n = all for t2star FT/S and for "
                "swi S, and 40 for swi FT. 'Not reached' therefore holds only for the intended reading: no "
                "plateau STRICTLY BELOW the full chain."),
       "per_sequence": {}}
for seq in ("t2star", "swi"):
    n_all = prim["n_all"][seq]
    seq_out = {"n_all": n_all, "variants": {}}
    for var in ("FT", "S"):
        ch = prim["targets"][seq]["chain"].get(var, {})
        per = ch.get("per_n", {})
        lb = {n: round(per[n]["lb95_F1n_minus_F1all"], 4) for n in per if n != str(n_all)}
        lb_all = per.get(str(n_all), {}).get("lb95_F1n_minus_F1all")
        largest_done = str(max(int(n) for n in lb))
        seq_out["variants"][var] = {
            "stored_plateau_n": ch.get("plateau_n"),
            "lb95_F1n_minus_F1all": lb,
            "largest_n_below_all": int(largest_done),
            "lb95_at_largest_n_below_all": lb[largest_done],
            "lb95_at_n_all": lb_all,
            "plateau_below_all_exists": any(v > MARGIN for v in lb.values()),
        }
    out["per_sequence"][seq] = seq_out

json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT, "-- plateau margin", MARGIN)
for seq, s in out["per_sequence"].items():
    print(seq, "n_all =", s["n_all"])
    for var, v in s["variants"].items():
        print("  %-2s stored plateau_n=%s  lb at n=%d: %+.4f   lb at n=all: %+.4f   plateau below all: %s"
              % (var, v["stored_plateau_n"], v["largest_n_below_all"], v["lb95_at_largest_n_below_all"],
                 v["lb95_at_n_all"], v["plateau_below_all_exists"]))
