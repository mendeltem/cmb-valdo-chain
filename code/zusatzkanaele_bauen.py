#!/usr/bin/env python3
"""T1 und T2 als Zusatzkanaele auf dem Trainingsgitter (Arm 8, 29.09.2026; Nutzerauftrag "T1-Kanal fuer VALDO").
Quelle: VALDO <id>_space-T2S_desc-masked_T1/T2 (im T2S-Raum, schaedelfrei; 33 von 72 T1 tragen NaN als Randschale -> nan_to_num).
Ziel: dev/gitter_d/<id>/<id>_t1.nii.gz, _t2.nii.gz -- linear auf Gitter und Zuschnitt des Falls gelegt (resample_from_to auf die
Affine/Form von <id>_image.nii.gz), im Hirn (image > 0) durch das 99.5-Perzentil geteilt und auf [0, 1] gekappt, ausserhalb 0.
Nicht invertiert (T1/T2 tragen keine Blutungshelligkeit, nur Anatomie). Prueft Form und Affine.
    python code/zusatzkanaele_bauen.py [--manifest dev/manifest_valdo_gitter_d.json] [--proc 6]
"""
import argparse, json, os
from multiprocessing import Pool
import nibabel as nib, numpy as np
from nibabel.processing import resample_from_to
ROOT = "/home/uchralt/data/work/valdo-t2s"; EXT = "/home/uchralt/data/extern/valdo2021/Task2"

def fall(e):
    sid, d = e["id"], e["dir"]; gi = nib.load(f"{d}/{sid}_image.nii.gz"); brain = np.asarray(gi.dataobj) > 0; out = {}
    for name, tag in (("t1", "T1"), ("t2", "T2")):
        src = nib.load(f"{EXT}/{sid}/{sid}_space-T2S_desc-masked_{tag}.nii.gz")
        a = np.nan_to_num(np.asarray(src.dataobj, dtype=np.float32), nan=0.0)
        rs = resample_from_to(nib.Nifti1Image(a, src.affine), (gi.shape, gi.affine), order=1)
        v = np.asarray(rs.dataobj, dtype=np.float32); v[~brain] = 0
        ref = float(np.percentile(v[brain], 99.5)) if brain.any() else 1.0
        v = np.clip(v / max(ref, 1e-6), 0, 1).astype(np.float32); v[brain & (v <= 0)] = 1e-6
        nib.save(nib.Nifti1Image(v, gi.affine, gi.header), f"{d}/{sid}_{name}.nii.gz")
        out[name] = dict(nan=int(np.isnan(np.asarray(src.dataobj, dtype=np.float32)).sum()), mean_brain=round(float(v[brain].mean()), 3))
    return sid, out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=f"{ROOT}/dev/manifest_valdo_gitter_d.json"); ap.add_argument("--proc", type=int, default=6)
    a = ap.parse_args(); cases = json.load(open(a.manifest))
    with Pool(a.proc) as pool:
        res = dict(pool.map(fall, cases))
    print(len(res), "Faelle; T1 mit NaN:", sum(r["t1"]["nan"] > 0 for r in res.values()), "| Mittel im Hirn T1 %.3f T2 %.3f" % (np.mean([r["t1"]["mean_brain"] for r in res.values()]), np.mean([r["t2"]["mean_brain"] for r in res.values()])))
    json.dump(res, open(f"{ROOT}/dev/zusatzkanaele_t1t2.json", "w"), indent=1)
