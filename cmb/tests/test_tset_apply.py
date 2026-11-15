"""cmb.stage2.tset_apply end to end on synthetic candidate folders (CPU, tiny training): runs, refuses overlapping cases,
writes the pre-registered test.

    python -m cmb.tests.test_tset_apply          (run from the repository root)
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cmb.tests.test_trackset import phantom      # noqa: E402


def make_set(folder, prefix, n_cases, rng):
    os.makedirs(folder, exist_ok=True)
    patches, rows, cases = [], [], {}
    for c in range(n_cases):
        cid = f"{prefix}{c:02d}"; n_ref = 0
        for j in range(3):
            bleed = (c + j) % 3 == 0
            patches.append(phantom("bleed" if bleed else "vessel", rng))
            rows.append(dict(case=cid, fold=c % 5, touched=[j + 1] if bleed else [], voxels=40, probability=0.5, tissue=2,
                             centre_voxel=[20 * j, 10, 5]))
            n_ref += bleed
        cases[cid] = dict(fold=c % 5, n_reference=n_ref)
    np.save(f"{folder}/patches.npy", np.stack(patches))
    json.dump(dict(threshold=0.15, candidates=rows, cases=cases), open(f"{folder}/candidates.json", "w"))


def test_confirmation_runs_and_writes_the_test():
    from cmb.stage2 import tset_apply
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as tmp:
        make_set(f"{tmp}/a", "a", 10, rng); make_set(f"{tmp}/b", "b", 10, rng); make_set(f"{tmp}/ext", "x", 6, rng)
        out = f"{tmp}/conf.json"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.argv = ["apply", "--cohort", f"a={tmp}/a", f"b={tmp}/b", "--target", f"ext={tmp}/ext", "--arm", "tset-track",
                    "--seed-offsets", "0", "--epochs", "2", "--members", "1", "--json", out]
        tset_apply.main()
        d = json.load(open(out))
        assert d["test"]["decision"] in ("confirmed", "not confirmed") and len(d["test"]["ci95"]) == 2
        assert set(d["runs"]) == {"tset-track", "tset-plain"} and "0" in d["runs"]["tset-track"]
        assert set(d["runs"]["tset-plain"]["0"]["per_case"]) == {f"x{c:02d}" for c in range(6)}
        assert set(d["reference"]) == {"probability", "probability+csf"}
        sys.argv = ["apply", "--cohort", f"a={tmp}/a", "--target", f"again={tmp}/a", "--arm", "tset-ctx", "--seed-offsets", "0",
                    "--epochs", "1", "--members", "1", "--json", out]
        try:
            tset_apply.main()
        except AssertionError as e:
            assert "not fresh" in str(e)
        else:
            raise AssertionError("accepted a target that overlaps the pool")


if __name__ == "__main__":
    test_confirmation_runs_and_writes_the_test()
    print("ok test_confirmation_runs_and_writes_the_test\nALL 1 TESTS OK")
