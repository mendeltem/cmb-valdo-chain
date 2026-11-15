"""wk5 (31.10.2026): pooled.write_decisions -- the lesion metrics recomputed from the dumped decisions equal scores_of."""
import json

import numpy as np

from cmb.stage2.pooled import scores_of, write_decisions


def _rows():
    rng = np.random.RandomState(0)
    rows = []
    for c in range(6):
        for j in range(8):
            touched = [f"L{j}"] if j < 3 else []                     # three reference lesions per case
            rows.append(dict(case=f"case{c}", touched=touched, voxels=int(rng.randint(3, 40)), tissue=1,
                             centre_voxel=[c, j, 0], probability=float(rng.rand())))
    return rows


def test_dump_recomputes_metrics(tmp_path):
    rows = _rows()
    n = len(rows)
    cohort = np.array(["a"] * 24 + ["b"] * 24)
    fold = np.array([i // 8 % 5 for i in range(n)])
    extra = np.zeros(n, bool); extra[-2:] = True                     # extra candidates must not be dumped
    rng = np.random.RandomState(1)
    kept = {"m1": rng.rand(n) > 0.4, "m2": rng.rand(n) > 0.6}
    dumps = {k: (rng.rand(n), np.full(n, 0.5)) for k in kept}
    path = tmp_path / "d.jsonl"
    lines = write_decisions(str(path), rows, cohort, fold, extra, kept, dumps, seed_offset=0)
    assert lines == 2 * (n - 2)
    recs = [json.loads(l) for l in open(path)]
    for mode, acc in kept.items():
        for name in ("a", "b"):
            sel = (cohort == name) & ~extra
            want = scores_of([r for r, s in zip(rows, sel) if s], acc[sel], 3 * len({r["case"] for r, s in zip(rows, sel) if s}),
                             len({r["case"] for r, s in zip(rows, sel) if s}))
            mine = [r for r in recs if r["mode"] == mode and r["cohort"] == name]
            got = scores_of([dict(case=r["case"], touched=r["touched"]) for r in mine], np.array([r["accepted"] for r in mine]),
                            3 * len({r["case"] for r in mine}), len({r["case"] for r in mine}))
            assert got == want
    assert all(r["score"] is not None and r["threshold"] == 0.5 for r in recs)
