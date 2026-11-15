"""cmb.analysis.fold_freeze on synthetic files: freeze, re-freeze guard, check finds a moved case and a tampered freeze.

    python -m cmb.tests.test_fold_freeze          (run from the repository root)
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


def run(*args):
    r = subprocess.run([PY, "-m", "cmb.analysis.fold_freeze", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout


def test_freeze_and_check():
    with tempfile.TemporaryDirectory() as tmp:
        json.dump(dict(folds=[[f"c{i}" for i in range(k * 3, k * 3 + 3)] for k in range(5)]), open(f"{tmp}/split.json", "w"))
        rc, out = run("freeze", "--cohort", "x", "--split", f"{tmp}/split.json", "--out", f"{tmp}/fz")
        assert rc == 0 and "frozen 15 cases" in out
        cases = {f"c{i}": dict(fold=i // 3, n_reference=0) for i in range(15)}
        os.makedirs(f"{tmp}/cand"); json.dump(dict(candidates=[], cases=cases), open(f"{tmp}/cand/candidates.json", "w"))
        os.makedirs(f"{tmp}/run/fold2"); open(f"{tmp}/run/fold2/c7_pred_proba.nii.gz", "w").close()
        rc, out = run("check", "--frozen", f"{tmp}/fz/x.json", "--candidates", f"{tmp}/cand", "--runs", f"{tmp}/run")
        assert rc == 0 and "FOLDS OK" in out, out
        cases["c4"]["fold"] = 3                                         # one case moved to another fold
        json.dump(dict(candidates=[], cases=cases), open(f"{tmp}/cand/candidates.json", "w"))
        rc, out = run("check", "--frozen", f"{tmp}/fz/x.json", "--candidates", f"{tmp}/cand")
        assert rc == 1 and "1 fold mismatches" in out
        bad = json.load(open(f"{tmp}/split.json")); bad["folds"][0], bad["folds"][1] = bad["folds"][1], bad["folds"][0]
        json.dump(bad, open(f"{tmp}/split2.json", "w"))
        rc, out = run("freeze", "--cohort", "x", "--split", f"{tmp}/split2.json", "--out", f"{tmp}/fz")
        assert rc == 1 and "REFUSED" in out                             # a different assignment is never written over the freeze
        d = json.load(open(f"{tmp}/fz/x.json")); d["folds"]["c0"] = 4; json.dump(d, open(f"{tmp}/fz/x.json", "w"))
        rc, out = run("check", "--frozen", f"{tmp}/fz/x.json")
        assert rc == 1 and "FREEZE CHANGED" in out


if __name__ == "__main__":
    test_freeze_and_check()
    print("ok test_freeze_and_check\nALL 1 TESTS OK")
