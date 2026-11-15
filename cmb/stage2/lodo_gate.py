"""wk3 HPC gate (06.11.2026): the UNCHANGED part of the LODO logic (cmb.stage2.lodo.nested_cv_score = III-A variant (a),
feature forest, seeds 0-2) must reproduce the stored the workstation forest result for every held-out site exactly (phase3a_forest
reference). This checks candidate sets, site split, twin rule (F1), folds and the threshold search on the target machine.
    python -m cmb.stage2.lodo_gate REFERENCE.json   -> exit 0 if all five sites are identical, else 1"""
import json, sys
from cmb.stage2 import lodo

ref = json.load(open(sys.argv[1]))
rows, x, cases, sites = lodo.load_all()
bad = 0
for D in sites:
    r = lodo.nested_cv_score(rows, x, cases, "forest", [0, 1, 2], {s for s in sites if s != D}, D)
    got = round(lodo.mean_ci(r)[0], 6); want = round(ref[D]["lodo_transfer"]["f1"], 6)
    print(f"{D}: LODO(a) forest F1 {got} reference {want} {'OK' if got == want else 'DIFFERENT'}", flush=True)
    bad += got != want
print("GATE", "OK" if not bad else "FAILED"); sys.exit(1 if bad else 0)
