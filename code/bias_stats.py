#!/usr/bin/env python3
"""we1: Biaskorrektur-Beleg -- Mittel/Std im Hirn vor und nach fast -B.

3 Beispielfaelle je Kohorte (so viele, wie schon fertig sind): Rohbild vs
fast_restore, Hirnmaske = Bild>0 des jeweiligen Bildes. Ausgabe nach
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
    # je Kohorte max. 3
    sel = {}
    for sid, v in out.items():
        sel.setdefault(v["kohorte"], [])
        if len(sel[v["kohorte"]]) < 3:
            sel[v["kohorte"]].append(sid)
    chosen = {s: out[s] for ss in sel.values() for s in ss}
    json.dump(chosen, open(f"/home/uchralt/data/work/valdo-t2s/dev/bias_stats.json", "w"),
              indent=1, ensure_ascii=False)
    for sid, v in sorted(chosen.items()):
        print(f"{sid} ko{v['kohorte']}  vor: {v['vor']['mittel']}+/-{v['vor']['std']} "
              f"({v['vor']['n_voxel']} vox)  nach: {v['nach']['mittel']}+/-{v['nach']['std']}")
    print("fertig:", len(chosen), "Faelle (mehr kommen, wenn mehr Faelle durch fast sind)")

if __name__ == "__main__":
    main()
