"""cmb.synthesis.insert_volume and cmb.synthesis.realism_probe on synthetic hosts (CPU, no patient data).

    python -m cmb.tests.test_insert_realism          (run from the repository root, ~3 min)
"""
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def host(tmp, cid, rng):
    d = f"{tmp}/hosts/{cid}"; os.makedirs(d)
    x, y, z = np.mgrid[0:96, 0:96, 0:48]
    brain = ((x - 48) * 0.5) ** 2 + ((y - 48) * 0.5) ** 2 + ((z - 24) * 1.0) ** 2 < 21 ** 2
    img = np.where(brain, 0.35 + 0.02 * rng.randn(96, 96, 48), 0).astype(np.float32)
    lab = np.zeros(img.shape, np.uint8); lab[48, 48, 24] = 1                       # one existing 'reference lesion'
    aff = np.diag([0.5, 0.5, 1.0, 1.0])
    for n, a in (("image", img), ("frst", np.zeros_like(img)), ("label", lab)):
        nib.save(nib.Nifti1Image(a, aff), f"{d}/{cid}_{n}.nii.gz")
    return dict(id=cid, dir=d, fold=int(cid[-1]) % 5)


def test_insert_volume_writes_consistent_copies():
    from cmb.synthesis import insert_volume
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as tmp:
        hosts = [host(tmp, f"h{k}", rng) for k in range(2)]
        json.dump(hosts, open(f"{tmp}/hosts.json", "w"))
        sys.argv = ["ins", "--manifest", f"{tmp}/hosts.json", "--b0", "3", "--te", "20", "--slice-mm", "3", "--inplane-mm", "0.5",
                    "--bleeds", "2", "--veins", "1", "--seed", "1", "--out", f"{tmp}/syn"]
        insert_volume.main()
        man = json.load(open(f"{tmp}/syn/manifest_synthetic.json")); log = json.load(open(f"{tmp}/syn/insert_log.json"))["objects"]
        assert [m["fold"] for m in man] == [h["fold"] for h in hosts] and all(m["synthetic"] for m in man)
        for h in hosts:
            cid = h["id"]; objs = log[cid]
            assert [o["kind"] for o in objs] == ["bleed", "bleed", "vein"]
            c = np.array([o["centre_voxel"] for o in objs]) * [0.5, 0.5, 1.0]
            assert min(np.linalg.norm(c[i] - c[j]) for i in range(3) for j in range(i)) >= 8.0
            assert all(np.linalg.norm(ci - np.array([24, 24, 24])) >= 8.0 for ci in c)          # away from the existing lesion
            old = np.asarray(nib.load(f"{h['dir']}/{cid}_image.nii.gz").dataobj); new = np.asarray(nib.load(f"{tmp}/syn/{cid}/{cid}_image.nii.gz").dataobj)
            lab = np.asarray(nib.load(f"{tmp}/syn/{cid}/{cid}_label.nii.gz").dataobj) > 0
            for o in objs:
                x, y, z = o["centre_voxel"]
                if o["kind"] == "bleed" and o["labelled"]:
                    assert lab[x - 3:x + 4, y - 3:y + 4, z - 2:z + 3].any() and new[x, y, z] > old[x, y, z]
                assert o["written"] == (o["kind"] == "vein" or o["labelled"])
            assert (new[old == 0] == 0).all() and lab[48, 48, 24]
            from scipy import ndimage
            n_new = ndimage.label(lab)[1] - 1
            assert n_new == sum(o["kind"] == "bleed" and o["labelled"] for o in objs)          # veins carry no label
            fr = np.asarray(nib.load(f"{tmp}/syn/{cid}/{cid}_frst.nii.gz").dataobj)
            assert fr.max() > 0 and (fr[old == 0] == 0).all()                                  # FRST recomputed


def test_faint_bleed_is_not_written():
    from cmb.synthesis import insert_volume
    rng = np.random.RandomState(3)
    img = np.full((60, 60, 30), 0.35, np.float32)
    weak = dict(b0=1.5, te=5, slice_mm=1.0, inplane_mm=0.5)
    real_sample = insert_volume.R.sample_object
    insert_volume.R.sample_object = lambda kind, rng, centre: dict(kind="sphere", centre=centre, radius=0.3, chi=0.05, w_core=1.0)
    try:
        before = img.copy()
        entry, apparent = insert_volume.render_into(img, np.array([30, 30, 15]), "bleed", rng, weak)
    finally:
        insert_volume.R.sample_object = real_sample
    assert entry["apparent_voxels"] == 0 and entry["written"] is False and np.array_equal(img, before)


def test_keep_inserted_filters_to_inserted_objects():
    from cmb.synthesis import keep_inserted
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f"{tmp}/c")
        rows = [dict(case="h0", fold=0, touched=[2], centre_voxel=[10, 10, 10]),     # at an inserted bleed -> keep
                dict(case="h0", fold=0, touched=[1], centre_voxel=[60, 60, 20]),     # the host's own lesion -> drop
                dict(case="h0", fold=0, touched=[], centre_voxel=[30, 30, 15]),      # at an inserted vein -> keep
                dict(case="h0", fold=0, touched=[], centre_voxel=[80, 10, 5])]       # host false alarm -> drop
        np.save(f"{tmp}/c/patches.npy", np.arange(4 * 2, dtype=np.float16).reshape(4, 2))
        json.dump(dict(candidates=rows, cases={"h0": dict(fold=0, n_reference=2)}), open(f"{tmp}/c/candidates.json", "w"))
        json.dump(dict(objects={"h0": [dict(kind="bleed", centre_voxel=[11, 10, 10], labelled=True),
                                       dict(kind="vein", centre_voxel=[30, 32, 15])]}), open(f"{tmp}/log.json", "w"))
        sys.argv = ["k", "--candidates", f"{tmp}/c", "--log", f"{tmp}/log.json", "--out", f"{tmp}/o"]
        keep_inserted.main()
        d = json.load(open(f"{tmp}/o/candidates.json")); p = np.load(f"{tmp}/o/patches.npy")
        assert [r["centre_voxel"] for r in d["candidates"]] == [[10, 10, 10], [30, 30, 15]] and p.tolist() == [[0, 1], [4, 5]]
        assert d["cases"] == {"h0": dict(fold=0, n_reference=2)}


def test_realism_probe_gates():
    from cmb.synthesis import realism_probe
    from cmb.tests.test_trackset import phantom
    rng = np.random.RandomState(2)
    with tempfile.TemporaryDirectory() as tmp:
        def folder(name, n_cases, synthetic):
            os.makedirs(f"{tmp}/{name}"); patches, rows, log = [], [], {}
            for c in range(n_cases):
                cid = f"c{c:02d}"; log[cid] = []
                for j, kind in enumerate(("bleed", "vessel", "bleed", "vessel")):
                    patches.append(phantom(kind, rng)); centre = [10 * j + 5, 10, 5]
                    rows.append(dict(case=cid, fold=c % 5, touched=[1] if kind == "bleed" else [], voxels=20, probability=0.5,
                                     centre_voxel=centre))
                    log[cid].append(dict(kind="bleed" if kind == "bleed" else "vein", centre_voxel=centre, labelled=True))
            np.save(f"{tmp}/{name}/patches.npy", np.stack(patches))
            json.dump(dict(candidates=rows, cases={}), open(f"{tmp}/{name}/candidates.json", "w"))
            if synthetic:
                json.dump(dict(objects=log), open(f"{tmp}/{name}/insert_log.json", "w"))
        folder("real", 20, False); folder("syn", 20, True)
        sys.argv = ["probe", "--real", f"x={tmp}/real", "--synthetic", f"x={tmp}/syn,{tmp}/syn/insert_log.json", "--epochs", "8",
                    "--json", f"{tmp}/r.json"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        realism_probe.main()
        r = json.load(open(f"{tmp}/r.json"))["cohorts"]["x"]
        assert r["A_bleeds"]["auc"] <= 0.75, r                    # same generator: indistinguishable (small-n slack)
        assert r["C_control"]["auc"] >= 0.85, r                   # the smoothed fake is caught
        assert r["gate"] in ("pass", "fail")


if __name__ == "__main__":
    for t in (test_insert_volume_writes_consistent_copies, test_faint_bleed_is_not_written, test_keep_inserted_filters_to_inserted_objects, test_realism_probe_gates):
        t(); print("ok", t.__name__, flush=True)
    print("ALL 4 TESTS OK")
