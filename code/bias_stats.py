#!/usr/bin/env python3
"""we1: bias correction evidence -- mean/std in the brain before and after fast -B.

3 example cases per cohort (as many as are already done): raw image vs
fast_restore, brain mask = image>0 of the respective image. Output to
dev/bias_stats.json.
"""
import os, json
import numpy as np, nibabel as nib

VALDO = "/home/uchralt/data/extern/valdo2021/Task2"
PREPROC = "/home/uchralt/data/work/valdo-t2s/dev/preproc"
PROBE = "/home/uchralt/data/work/valdo-t2s/dev/preproc-probe"

def stats_of(path):
    img = nib.load(path).get_fdata()
    m = img > 0
    return {"n_voxel": int(m.sum()),
            "mittel": round(float(img[m].mean()), 4),
            "std": round(float(img[m].std()), 4)}

def main():
    out = {}
    for sid in sorted(os.listdir(VALDO)):
        d = f"{VALDO}/{sid}"
        raw = f"{d}/{sid}_space-T2S_desc-masked_T2S.nii.gz"
        restore = f"{PREPROC}/{sid}/fast_restore.nii.gz"
        if not os.path.exists(restore):
            alt = f"{PROBE}/{sid}/fast_restore.nii.gz"
            if os.path.exists(alt):
                restore = alt
            else:
                continue
        out[sid] = {"kohorte": sid[4],
                    "vor": stats_of(raw),
                    "nach": stats_of(restore)}
    # max. 3 per cohort
    sel = {}
    for sid, v in out.items():
        sel.setdefault(v["kohorte"], [])
        if len(sel[v["kohorte"]]) < 3:
            sel[v["kohorte"]].append(sid)
    chosen = {s: out[s] for ss in sel.values() for s in ss}
    json.dump(chosen, open(f"/home/uchralt/data/work/valdo-t2s/dev/bias_stats.json", "w"),
              indent=1, ensure_ascii=False)
    for sid, v in sorted(chosen.items()):
        print(f"{sid} ko{v['kohorte']}  before: {v['vor']['mittel']}+/-{v['vor']['std']} "
              f"({v['vor']['n_voxel']} vox)  after: {v['nach']['mittel']}+/-{v['nach']['std']}")
    print("done:", len(chosen), "cases (more will come once more cases have gone through fast)")

if __name__ == "__main__":
    main()
