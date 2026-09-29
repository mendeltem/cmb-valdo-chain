"""Empirical size and contrast distribution of the REAL VALDO microbleeds on the training grid (2026-09-28).

Why: synthetic lesions must be drawn from the real distribution, not invented. Per lesion (26-connected component of the
label on gitter_d): volume in mm3, mean contrast (mean inside - mean of a 0.5-2 mm in-plane ring), peak contrast
(max inside - ring mean), both on the stored inverted scale (1 - I/max, lesions bright), and the SynthSeg tissue class
under the lesion (1 CSF, 2 cortex, 3 white matter, 4 deep nuclei, 5 infratentorial), plus the z extent in slices.

    python -m cmb.synthesis.lesion_stats --manifest dev/manifest_valdo_gitter_d.json --json out.json
"""
import argparse, json, os
import nibabel as nib, numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
CLASSES = {}
for l in (4, 5, 14, 15, 24, 43, 44): CLASSES[l] = 1
for l in (3, 42, 17, 18, 53, 54): CLASSES[l] = 2
for l in (2, 41): CLASSES[l] = 3
for l in (10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60): CLASSES[l] = 4
for l in (7, 8, 16, 46, 47): CLASSES[l] = 5


def per_case(entry, synthseg):
    d, sid = entry["dir"], entry["id"]
    ni = nib.load(f"{d}/{sid}_image.nii.gz"); img = np.asarray(ni.dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(f"{d}/{sid}_label.nii.gz").dataobj) > 0
    seg = np.asarray(nib.load(synthseg.format(id=sid)).dataobj).astype(np.int32)
    sp = ni.header.get_zooms()[:3]; vox = float(np.prod(sp))
    ll, n = ndimage.label(lab, structure=np.ones((3, 3, 3)))
    rows = []
    # in-plane ring 0.5-2 mm: dilate in x/y only
    r_in, r_out = int(round(0.5 / sp[0])), int(round(2.0 / sp[0]))
    disk = lambda r: (lambda y, x: (x - r) ** 2 + (y - r) ** 2 <= r * r)(*np.ogrid[:2 * r + 1, :2 * r + 1])
    for j in range(1, n + 1):
        m = ll == j
        idx = np.argwhere(m); lo = idx.min(0) - r_out - 1; hi = idx.max(0) + r_out + 2
        lo = np.maximum(lo, 0); hi = np.minimum(hi, m.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        mm, im = m[sl], img[sl]
        outer = np.zeros_like(mm); inner = np.zeros_like(mm)
        for z in range(mm.shape[2]):
            if mm[:, :, z].any():
                outer[:, :, z] = ndimage.binary_dilation(mm[:, :, z], structure=disk(r_out))
                inner[:, :, z] = ndimage.binary_dilation(mm[:, :, z], structure=disk(r_in))
        ring = outer & ~inner & (im > 0) & ~(lab[sl])
        if ring.sum() < 5:
            continue
        rm = float(im[ring].mean())
        vals, counts = np.unique(seg[sl][mm], return_counts=True)
        cls = CLASSES.get(int(vals[np.argmax(counts)]), 0)
        rows.append(dict(case=sid, cohort=entry["cohort"], volume_mm3=round(float(mm.sum() * vox), 2),
                         z_slices=int(idx[:, 2].max() - idx[:, 2].min() + 1),
                         mean_contrast=round(float(im[mm].mean() - rm), 4), peak_contrast=round(float(im[mm].max() - rm), 4),
                         ring_mean=round(rm, 4), tissue_class=cls))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--synthseg", default="dev/synthseg/{id}_synthseg.nii.gz")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    syn = a.synthseg if os.path.isabs(a.synthseg) else f"{ROOT}/{a.synthseg}"
    rows = [r for e in json.load(open(a.manifest)) for r in per_case(e, syn)]
    def q(x): return [round(float(v), 3) for v in np.percentile(x, [10, 25, 50, 75, 90])]
    summary = {}
    for coh in sorted({r["cohort"] for r in rows}) + ["all"]:
        rs = [r for r in rows if coh == "all" or r["cohort"] == coh]
        summary[coh] = dict(n=len(rs), volume_mm3_p=q([r["volume_mm3"] for r in rs]), mean_contrast_p=q([r["mean_contrast"] for r in rs]),
                            peak_contrast_p=q([r["peak_contrast"] for r in rs]), z_slices_p=q([r["z_slices"] for r in rs]),
                            tissue_class_counts={str(c): sum(r["tissue_class"] == c for r in rs) for c in range(6)})
    json.dump(dict(summary=summary, lesions=rows), open(a.json, "w"), indent=1)
    for coh, s in summary.items():
        print(coh, s)


if __name__ == "__main__":
    main()
