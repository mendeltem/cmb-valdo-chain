"""Lesion volumes in ml per case: reference and every run's prediction (fixed cell 0.3 / >= 2 mm3 / giant removed; optional CSF
rule), like ``fslmaths -V`` (voxel count x voxel volume) but on the training grid and after the project's post-processing.
Writes a CSV (one row per case) and prints the totals; user request 2026-09-29.

    python -m cmb.analysis.volumes --run NAME=FOLDER[:csf] ... --manifest M.json [--synthseg PATTERN] --csv out.csv
Private cohorts: private --csv.
"""
from __future__ import annotations
import argparse, csv, glob, json, os
import nibabel as nib, numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
N26 = np.ones((3, 3, 3), bool); T, MIN_MM3, MAX_VOX, VOX = 0.3, 2.0, 2100, 0.25
CSF = {4, 5, 14, 15, 24, 43, 44}


def post_process(prob, seg):
    labels, n = ndimage.label(prob > T, structure=N26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOX >= MIN_MM3) & (size <= MAX_VOX); keep[0] = False
    if seg is not None:
        for c, box in enumerate(ndimage.find_objects(labels), start=1):
            if keep[c]:
                v, k = np.unique(seg[box][labels[box] == c], return_counts=True)
                if int(v[np.argmax(k)]) in CSF:
                    keep[c] = False
    return keep[labels]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True); ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--synthseg"); ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    runs = []
    for spec in a.run:
        name, folder = spec.split("=", 1); csf = folder.endswith(":csf"); folder = folder[:-4] if csf else folder
        runs.append((name, absolute(folder), csf))
    cases = [c for m in a.manifest for c in json.load(open(absolute(m)))]
    rows = []; tot = {"referenz_ml": 0.0, **{n: 0.0 for n, _, _ in runs}}
    for c in cases:
        cid, d = c["id"], c["dir"]
        gi = nib.load(f"{d}/{cid}_image.nii.gz"); vox_ml = float(np.prod(gi.header.get_zooms()[:3])) / 1000.0
        lab = np.asarray(nib.load(f"{d}/{cid}_label.nii.gz").dataobj) > 0
        seg = None
        if a.synthseg and any(r[2] for r in runs):
            si = nib.load(absolute(a.synthseg).format(id=cid, sid=cid.split("-", 1)[1] if "-" in cid else cid))
            if si.shape != gi.shape:
                from nibabel.processing import resample_from_to
                si = resample_from_to(si, (gi.shape, gi.affine), order=0)
            seg = np.asarray(si.dataobj).astype(np.int32)
        row = dict(fall=cid, kohorte=c.get("cohort", ""), referenz_n=int(ndimage.label(lab, structure=N26)[1]), referenz_ml=round(float(lab.sum()) * vox_ml, 4))
        tot["referenz_ml"] += row["referenz_ml"]
        for name, folder, csf in runs:
            found = glob.glob(f"{folder}/fold*/{cid}_pred_proba.nii.gz")
            if not found:
                row[name + "_n"] = ""; row[name + "_ml"] = ""; continue
            pred = post_process(np.asarray(nib.load(found[0]).dataobj).astype(np.float32) / 255.0, seg if csf else None)
            row[name + "_n"] = int(ndimage.label(pred, structure=N26)[1]); row[name + "_ml"] = round(float(pred.sum()) * vox_ml, 4)
            tot[name] += row[name + "_ml"]
        rows.append(row); print(f"{cid}: Referenz {row['referenz_n']} / {row['referenz_ml']:.4f} ml | " + " | ".join(f"{n} {row[n + '_n']} / {row[n + '_ml']} ml" for n, _, _ in runs), flush=True)
    out = absolute(a.csv); os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSummen ({len(rows)} Faelle): Referenz {tot['referenz_ml']:.3f} ml | " + " | ".join(f"{n} {tot[n]:.3f} ml" for n, _, _ in runs) + f"\n-> {out}")


if __name__ == "__main__":
    main()
