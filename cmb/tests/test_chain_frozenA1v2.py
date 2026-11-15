"""CPU unit test for MB/paper/phase4/chain_frozenA1v2.py on a SYNTHETIC block (no real data, exit 0).

Builds two tiny synthetic candidate folders (a VALDO-like pool and one phase-2-like target
block) with the phantom patches of cmb.tests.test_trackset, points selection.json at a
temporary winner arm (all three implementable arms are tested), and checks that
valdo_only_classifier + score_group produce finite scores, thresholds in THRESHOLDS and
per_case triples that add up to the reference counts.

    python cmb/tests/test_chain_frozenA1v2.py   (from the repository root; needs CMB_PRIVATE)
"""
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from cmb.tests.test_trackset import phantom      # noqa: E402
from cmb.stage2.classifier import THRESHOLDS     # noqa: E402

P4 = __import__("cmb.paths").paths.PRIVATE + "/paper/phase4"
SPEC = importlib.util.spec_from_file_location("chain_frozenA1v2", f"{P4}/chain_frozenA1v2.py")
CF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CF)


def make_set(folder, prefix, n_cases, rng, n_ref_every=2):
    os.makedirs(folder, exist_ok=True)
    patches, rows, cases = [], [], {}
    for c in range(n_cases):
        cid = f"{prefix}{c:02d}"; n_ref = 0
        for j in range(4):
            bleed = (c + j) % n_ref_every == 0
            patches.append(phantom("bleed" if bleed else "vessel", rng))
            rows.append(dict(case=cid, fold=c % 5, touched=[j + 1] if bleed else [], voxels=40,
                             probability=0.5, tissue=2, centre_voxel=[20 * j, 10, 5]))
            n_ref += 1 if bleed else 0
        cases[cid] = dict(fold=c % 5, n_reference=n_ref)
    np.save(f"{folder}/patches.npy", np.stack(patches).astype(np.float16))
    json.dump(dict(threshold=0.15, candidates=rows, cases=cases), open(f"{folder}/candidates.json", "w"))


def test_arm_end_to_end(arm: str):
    rng = np.random.RandomState(7)
    with tempfile.TemporaryDirectory() as tmp:
        pool_dir, tgt_dir = f"{tmp}/pool", f"{tmp}/tgt"
        make_set(pool_dir, "p", 10, rng)
        make_set(tgt_dir, "t", 4, rng)
        sel = f"{tmp}/selection.json"
        json.dump(dict(winner=arm), open(sel, "w"))
        CF.SELECTION = sel
        CF.VALDO_POOL = pool_dir
        # small budgets: this is a unit test on phantoms, not a training run
        rows, fit, inputs, thr, per_cohort = CF.valdo_only_classifier(arm, 0, epochs=2, members=1)
        assert thr in [float(t) for t in THRESHOLDS]
        assert all(t in [float(t2) for t2 in THRESHOLDS] for t in per_cohort.values())
        assert CF.winner_arm() == arm
        res = CF.score_group([tgt_dir], 0, arm, fit, inputs, thr, per_cohort)
        r = res[tgt_dir]
        assert "error" not in r and not r.get("empty")
        for label in ("a_pooled", "b_percohort"):
            m = r[label]
            for c, (tp, fp, fn) in m["per_case"].items():
                assert tp + fn == 2  # n_ref_every=2 -> two reference lesions per case
            tot_tp = sum(v[0] for v in m["per_case"].values())
            tot_fp = sum(v[1] for v in m["per_case"].values())
            tot_fn = sum(v[2] for v in m["per_case"].values())
            expect_f1 = 2 * tot_tp / max(2 * tot_tp + tot_fp + tot_fn, 1)
            assert abs(m["f1"] - expect_f1) < 1e-9, (label, m["f1"], expect_f1)
        print(f"  {arm}: a_pooled F1 {r['a_pooled']['f1']:.3f}, b_percohort F1 {r['b_percohort']['f1']:.3f} -- ok")


def test_empty_block_is_a_result():
    """A block without patches.npy (stage 1 found nothing) must give F1 0, not crash."""
    rng = np.random.RandomState(11)
    with tempfile.TemporaryDirectory() as tmp:
        pool_dir, tgt_dir = f"{tmp}/pool", f"{tmp}/empty"
        make_set(pool_dir, "p", 10, rng)
        os.makedirs(tgt_dir)
        json.dump(dict(threshold=0.15, candidates=[], cases={}), open(f"{tgt_dir}/candidates.json", "w"))
        CF.VALDO_POOL = pool_dir
        rows, fit, inputs, thr, per_cohort = CF.valdo_only_classifier("feat-vessel", 0, epochs=2, members=1)
        CF.SELECTION = f"{tmp}/selection.json"
        json.dump(dict(winner="feat-vessel"), open(CF.SELECTION, "w"))
        res = CF.score_group([tgt_dir], 0, "feat-vessel", fit, inputs, thr, per_cohort)
        r = res[tgt_dir]
        assert r["empty"] and r["a_pooled"]["f1"] == 0.0 and r["b_percohort"]["f1"] == 0.0
        print("  empty block: F1 0, no error -- ok")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for arm in ("feat-vessel", "feat-vessel-tissue", "tset-plain"):
        test_arm_end_to_end(arm)
    test_empty_block_is_a_result()
    print("test_chain_frozenA1v2: all passed")
