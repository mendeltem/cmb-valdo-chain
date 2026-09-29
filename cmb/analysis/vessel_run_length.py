"""Arm 5 (WORKLOG.md 5ae, 2026-09-29): are false alarms vessels running through many slices?  Dark run length along z.

For every candidate (component of the out-of-fold probability map at 0.15 / >= 1 mm3 -- the second stage's candidates):
  * z_extent_mm      : through-plane extent of the thresholded component itself;
  * run_length_mm    : from the candidate centre follow, slice by slice up and down, the local maximum of the inverted
                       image (microbleeds and vessels bright) inside a +-1 mm window (oblique vessels), as long as the
                       contrast to an in-plane ring (1-2 mm) stays >= half of the contrast at the centre; length = up + down + 1;
  * touched          : reference lesions hit (touch rule, 26-connectivity) -> true / false candidate.
Then: distributions true vs false (percentiles, AUC) and the rule "run_length > L -> reject" with L chosen per fold on
the OTHER folds (nested), applied on top of the fixed cell (0.3 / 2 mm3), paired against the fixed cell and against
fixed cell + CSF rule (bootstrap over cases). Per-candidate tables only leave the machine for public cohorts.

    python -m cmb.analysis.vessel_run_length --run NAME=FOLDER --manifest M.json [--synthseg PATTERN] --json out.json
"""
from __future__ import annotations
import argparse, glob, json, os
from multiprocessing import Pool
import nibabel as nib, numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
N26 = np.ones((3, 3, 3), bool)
CAND_T, CAND_MM3, FIX_T, FIX_MM3, MAX_VOX, VOX = 0.15, 1.0, 0.3, 2.0, 2100, 0.25
CSF = {4, 5, 14, 15, 24, 43, 44}
L_GRID = [6, 8, 10, 12, 15, 20]
WIN, RING_IN, RING_OUT = 2, 2, 4          # voxels in-plane (0.5 mm): +-1 mm window, ring 1-2 mm


def ring_mean(img, x, y, z):
    sl = img[max(x - RING_OUT, 0):x + RING_OUT + 1, max(y - RING_OUT, 0):y + RING_OUT + 1, z]
    gx, gy = np.ogrid[max(x - RING_OUT, 0):x + RING_OUT + 1, max(y - RING_OUT, 0):y + RING_OUT + 1]
    d2 = (gx - x) ** 2 + (gy - y) ** 2
    m = (d2 > RING_IN ** 2) & (d2 <= RING_OUT ** 2) & (sl > 0)
    return float(sl[m].mean()) if m.sum() >= 4 else None


def follow(img, brain, x, y, z, ref_contrast, direction):
    n = 0
    while True:
        z += direction
        if z < 0 or z >= img.shape[2]:
            break
        sl = img[max(x - WIN, 0):x + WIN + 1, max(y - WIN, 0):y + WIN + 1, z]
        if sl.size == 0 or not (sl > 0).any():
            break
        i = np.unravel_index(np.argmax(sl), sl.shape)
        nx, ny = max(x - WIN, 0) + i[0], max(y - WIN, 0) + i[1]
        rm = ring_mean(img, nx, ny, z)
        if rm is None or img[nx, ny, z] - rm < 0.5 * ref_contrast:
            break
        x, y = nx, ny; n += 1
        if n > 200:
            break
    return n


def shape_scores(img, box_lo, box_hi, m_box_offset, comp_mask, sp):
    """Frangi vesselness (bright tubes) and blobness at the candidate, on a LOCAL box resampled to isotropic 0.5 mm.
    Eigenvalues |l1| <= |l2| <= |l3| of the Hessian (bright structure: l2, l3 < 0). Frangi: Ra = |l2|/|l3| (plate vs line),
    Rb = |l1|/sqrt(|l2 l3|) (blob vs line), S = norm. Vessel-like: Rb small; blob-like: Rb ~ 1. We report, inside the
    candidate component dilated by 1 mm, the maximum Frangi vesselness over sigmas 1-3 voxels (0.5-1.5 mm) and the
    maximum blobness = Rb * (1 - exp(-S^2 / 2c^2)) at the same scales (both in [0, 1])."""
    from scipy.ndimage import zoom, binary_dilation
    from skimage.filters import frangi
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    sub = img[box_lo[0]:box_hi[0], box_lo[1]:box_hi[1], box_lo[2]:box_hi[2]].astype(np.float32)
    fz = sp[2] / sp[0]
    iso = zoom(sub, (1, 1, fz), order=1) if abs(fz - 1) > 1e-3 else sub
    mask = np.zeros(sub.shape, bool); mask[m_box_offset] = comp_mask
    mask = zoom(mask.astype(np.uint8), (1, 1, fz), order=0).astype(bool) if abs(fz - 1) > 1e-3 else mask
    mask = binary_dilation(mask, iterations=2)
    ves = frangi(iso, sigmas=(1, 2, 3), black_ridges=False, alpha=0.5, beta=0.5, gamma=None)
    blob = np.zeros(iso.shape, np.float32)
    for sg in (1, 2, 3):
        H = hessian_matrix(iso, sigma=sg, order="rc", use_gaussian_derivatives=False)
        ev = hessian_matrix_eigvals(H)                      # sorted descending: ev[0] >= ev[1] >= ev[2]
        a = np.sort(np.abs(ev), axis=0)                      # |l1| <= |l2| <= |l3|
        bright = (ev[1] < 0) & (ev[2] < 0)                   # two most negative curvatures -> bright structure
        S = np.sqrt((ev ** 2).sum(0)); c = 0.5 * S.max() if S.max() > 0 else 1.0
        rb = a[0] / np.sqrt(a[1] * a[2] + 1e-12)
        b = np.where(bright, rb * (1 - np.exp(-(S ** 2) / (2 * c ** 2))), 0)
        blob = np.maximum(blob, b.astype(np.float32))
    return float(ves[mask].max()) if mask.any() else 0.0, float(blob[mask].max()) if mask.any() else 0.0


def one_case(job):
    entry, folder, synthseg = job
    cid, d = entry["id"], entry["dir"]
    load = lambda p: np.asarray(nib.load(p).dataobj)
    img = load(f"{d}/{cid}_image.nii.gz").astype(np.float32)
    ref = load(f"{d}/{cid}_label.nii.gz") > 0
    found = glob.glob(f"{folder}/fold*/{cid}_pred_proba.nii.gz")
    if not found:
        return []
    prob = load(found[0]).astype(np.float32) / 255.0
    sp = nib.load(f"{d}/{cid}_image.nii.gz").header.get_zooms()[:3]; dz = float(sp[2])
    seg = None
    if synthseg:
        si = nib.load(synthseg.format(id=cid, sid=cid.split("-", 1)[1] if "-" in cid else cid))
        if si.shape != img.shape:
            from nibabel.processing import resample_from_to
            si = resample_from_to(si, (img.shape, nib.load(f"{d}/{cid}_image.nii.gz").affine), order=0)
        seg = np.asarray(si.dataobj).astype(np.int32)
    rl, n_ref = ndimage.label(ref, structure=N26)
    labels, n = ndimage.label(prob > CAND_T, structure=N26)
    brain = img > 0
    edt = ndimage.distance_transform_edt(brain, sampling=sp)      # distance to the brain-mask boundary (Sundaresan 2023: < 5 mm -> reject)
    rows = []
    for c, box in enumerate(ndimage.find_objects(labels), start=1):
        m = labels[box] == c
        vox = int(m.sum())
        if vox * VOX < CAND_MM3:
            continue
        pmax = float(prob[box][m].max())
        idx = np.argwhere(m); ctr = idx[np.argmax(prob[box][m])] + np.array([s.start for s in box])
        x, y, z = (int(v) for v in ctr)
        touched = sorted(int(v) for v in np.unique(rl[box][m]) if v)
        rm = ring_mean(img, x, y, z)
        contrast = float(img[x, y, z] - rm) if rm is not None else 0.0
        if contrast > 0:
            up = follow(img, brain, x, y, z, contrast, +1); down = follow(img, brain, x, y, z, contrast, -1)
        else:
            up = down = 0
        # --- global component geometry (2026-09-29, literature: bounding-box / PCA axis ratio, van den Heuvel 2016; slice consistency)
        pts = (idx + np.array([b.start for b in box])) * np.array(sp)          # voxel coords -> mm
        if len(pts) >= 3:
            ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]; ev = np.clip(ev, 1e-6, None)
            pca_ratio = float(np.sqrt(ev[0] / ev[-1]))
        else:
            pca_ratio = 1.0
        zs = np.unique(idx[:, 2]); ratios2d = []; cents = []
        for zz in zs:
            q = pts[idx[:, 2] == zz][:, :2]
            cents.append(q.mean(0))
            if len(q) >= 3:
                e2 = np.sort(np.linalg.eigvalsh(np.cov(q.T)))[::-1]; e2 = np.clip(e2, 1e-6, None); ratios2d.append(float(np.sqrt(e2[0] / e2[1])))
        inplane_ratio = max(ratios2d) if ratios2d else 1.0
        shift = max((float(np.linalg.norm(cents[i + 1] - cents[i])) for i in range(len(cents) - 1)), default=0.0)
        src_thick = float(json.load(open(f"{d}/info.json")).get("quelle_spacing", [0, 0, dz])[2]) if os.path.exists(f"{d}/info.json") else dz
        n_blocks = int(round(float(np.ptp(idx[:, 2]) + 1) * dz / max(src_thick, dz) + 0.499))
        pad = np.array([16, 16, 8])                          # 8 mm box around the component for the shape scores
        lo = np.maximum(np.array([b.start for b in box]) - pad, 0); hi = np.minimum(np.array([b.stop for b in box]) + pad, img.shape)
        off = tuple(slice(b.start - l, b.stop - l) for b, l in zip(box, lo))
        try:
            vess, blobness = shape_scores(img, lo, hi, off, m, sp)
        except Exception as ex:                              # never lose a candidate over the extra feature
            vess, blobness = -1.0, -1.0
        csf = False
        if seg is not None:
            vals, counts = np.unique(seg[box][m], return_counts=True); csf = int(vals[np.argmax(counts)]) in CSF
        rows.append(dict(case=cid, fold=entry["fold"], touched=touched, voxels=vox, p_max=pmax, in_csf=csf, contrast=round(contrast, 4),
                         centre=[x, y, z], border_mm=round(float(edt[x, y, z]), 2),
                         z_extent_mm=round(float(np.ptp(idx[:, 2]) + 1) * dz, 1), run_length_mm=round((up + down + 1) * dz, 1),
                         vesselness=round(vess, 4), blobness=round(blobness, 4),
                         pca_ratio=round(pca_ratio, 3), inplane_ratio=round(inplane_ratio, 3), centroid_shift_mm=round(shift, 2), n_blocks=n_blocks,
                         fixed=bool(pmax >= FIX_T and vox * VOX >= FIX_MM3 and vox <= MAX_VOX)))
    return rows


def scores(rows, acc, n_ref, n_cases):
    found = {(r["case"], l) for r, a in zip(rows, acc) if a for l in r["touched"]}
    fp = sum(1 for r, a in zip(rows, acc) if a and not r["touched"]); tp = len(found); fn = n_ref - tp
    return dict(f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), sensitivity=tp / max(tp + fn, 1), fp_per_case=fp / max(n_cases, 1), tp=tp, fp=fp, fn=fn)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="NAME=FOLDER"); ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--synthseg"); ap.add_argument("--json"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--table", help="private per-candidate table (json)")
    a = ap.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    name, folder = a.run.split("=", 1); folder = absolute(folder)
    cases = [c for m in a.manifest for c in json.load(open(absolute(m)))]
    n_ref = sum(c["n_cmb"] for c in cases); n_cases = len(cases)
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(one_case, [(c, folder, absolute(a.synthseg) if a.synthseg else None) for c in cases]) for r in rs]
    fold = np.array([r["fold"] for r in rows]); fixed = np.array([r["fixed"] for r in rows]); csf = np.array([r["in_csf"] for r in rows])
    run = np.array([r["run_length_mm"] for r in rows]); zext = np.array([r["z_extent_mm"] for r in rows]); true = np.array([bool(r["touched"]) for r in rows])
    vess = np.array([r["vesselness"] for r in rows]); blob = np.array([r["blobness"] for r in rows]); vb = vess - blob
    border = np.array([r["border_mm"] for r in rows])
    pca = np.array([r["pca_ratio"] for r in rows]); inpl = np.array([r["inplane_ratio"] for r in rows]); shift = np.array([r["centroid_shift_mm"] for r in rows]); nbl = np.array([r["n_blocks"] for r in rows])
    out = dict(run=name, candidates=len(rows), cases=n_cases, reference=n_ref)
    q = lambda v: [round(float(x), 1) for x in np.percentile(v, [10, 25, 50, 75, 90])] if len(v) else None
    for label, sel in (("all", np.ones(len(rows), bool)), ("fixed_cell", fixed)):
        out[label] = dict(n_true=int((true & sel).sum()), n_false=int((~true & sel).sum()),
                          run_true_p=q(run[true & sel]), run_false_p=q(run[~true & sel]), auc_run=auc(run[~true & sel], run[true & sel]),
                          zext_true_p=q(zext[true & sel]), zext_false_p=q(zext[~true & sel]), auc_zext=auc(zext[~true & sel], zext[true & sel]))
        out[label]["auc_border"] = auc(-border[~true & sel], -border[true & sel])    # false closer to the boundary?
        out[label].update(auc_pca=auc(pca[~true & sel], pca[true & sel]), auc_inplane=auc(inpl[~true & sel], inpl[true & sel]),
                          auc_shift=auc(shift[~true & sel], shift[true & sel]), auc_nblocks=auc(nbl[~true & sel], nbl[true & sel]),
                          pca_true_p=q(pca[true & sel]), pca_false_p=q(pca[~true & sel]), shift_true_p=q(shift[true & sel]), shift_false_p=q(shift[~true & sel]))
        print(f"   geometry: PCA-ratio AUC(false more elongated) {out[label]['auc_pca']:.3f} (true p10/50/90 {out[label]['pca_true_p'][::2]}, false {out[label]['pca_false_p'][::2]}) | in-plane ratio AUC {out[label]['auc_inplane']:.3f} | centroid shift AUC {out[label]['auc_shift']:.3f} (true {out[label]['shift_true_p'][::2]}, false {out[label]['shift_false_p'][::2]}) | n_blocks AUC {out[label]['auc_nblocks']:.3f}")
        out[label]["border_true_p"] = q(border[true & sel]); out[label]["border_false_p"] = q(border[~true & sel])
        out[label].update(auc_vesselness=auc(vess[~true & sel], vess[true & sel]), auc_blobness=auc(blob[true & sel], blob[~true & sel]), auc_vess_minus_blob=auc(vb[~true & sel], vb[true & sel]),
                          vess_true_p=[round(float(x), 3) for x in np.percentile(vess[true & sel], [10, 50, 90])] if (true & sel).any() else None,
                          vess_false_p=[round(float(x), 3) for x in np.percentile(vess[~true & sel], [10, 50, 90])] if (~true & sel).any() else None)
        print(f"{name} [{label}] true {out[label]['n_true']} false {out[label]['n_false']} | run mm true {out[label]['run_true_p']} false {out[label]['run_false_p']} AUC(false longer) {out[label]['auc_run']:.3f} | z-extent AUC {out[label]['auc_zext']:.3f}"
              f" | border mm true {out[label]['border_true_p']} false {out[label]['border_false_p']} AUC(false closer) {out[label]['auc_border']:.3f}"
              f" | Frangi AUC(false more vessel-like) {out[label]['auc_vesselness']:.3f} (true p10/50/90 {out[label]['vess_true_p']}, false {out[label]['vess_false_p']}) | blobness AUC(true blobbier) {out[label]['auc_blobness']:.3f} | vess-blob AUC {out[label]['auc_vess_minus_blob']:.3f}")
    # rule, nested over folds
    def nested(base, feat=None, grid=None):
        feat = run if feat is None else feat; grid = L_GRID if grid is None else grid
        acc = base.copy()
        chosen = {}
        for k in sorted(set(fold)):
            others = fold != k
            best = max(grid, key=lambda L: scores([r for r, o in zip(rows, others) if o], (base & (feat <= L))[others], sum(c["n_cmb"] for c in cases if c["fold"] != k), sum(1 for c in cases if c["fold"] != k))["f1"])
            chosen[int(k)] = best; acc[fold == k] = base[fold == k] & (feat[fold == k] <= best)
        return acc, chosen
    per_case = lambda acc: {c["id"]: (len({l for r, a in zip(rows, acc) if a and r["case"] == c["id"] for l in r["touched"]}),
                                     sum(1 for r, a in zip(rows, acc) if a and r["case"] == c["id"] and not r["touched"]), c["n_cmb"]) for c in cases}
    def paired(A, B, rng):
        ids = list(A); f1 = lambda tr: 2 * sum(t[0] for t in tr) / max(2 * sum(t[0] for t in tr) + sum(t[1] for t in tr) + sum(t[2] - t[0] for t in tr), 1)
        d = []
        for _ in range(5000):
            ix = rng.randint(len(ids), size=len(ids)); d.append(f1([A[ids[i]] for i in ix]) - f1([B[ids[i]] for i in ix]))
        return round(f1(list(A.values())) - f1(list(B.values())), 3), [round(float(v), 3) for v in np.percentile(d, [2.5, 97.5])]
    rng = np.random.RandomState(0)
    out["rules"] = {}
    V_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]                      # vesselness thresholds (1.01 = rule off)
    for base_name, base, feat, grid in (("fixed", fixed, run, L_GRID), ("fixed+csf", fixed & ~csf, run, L_GRID),
                                        ("fixed | frangi", fixed, vess, V_GRID), ("fixed+csf | frangi", fixed & ~csf, vess, V_GRID),
                                        ("fixed+csf | vess-blob", fixed & ~csf, vb, [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 1.01]),
                                        ("fixed+csf | pca-ratio", fixed & ~csf, pca, [1.5, 2.0, 2.5, 3.0, 4.0, 99]),            # keep if ratio <= t
                                        ("fixed+csf | inplane", fixed & ~csf, inpl, [1.5, 2.0, 2.5, 3.0, 4.0, 99]),
                                        ("fixed+csf | shift", fixed & ~csf, shift, [0.5, 1.0, 1.5, 2.0, 3.0, 99]),
                                        ("fixed | border", fixed, -border, [-7, -5, -4, -3, -2, 0.01]),           # keep if border >= d (d = 7..2 mm; 0.01 = off)
                                        ("fixed+csf | border", fixed & ~csf, -border, [-7, -5, -4, -3, -2, 0.01])):
        acc, chosen = nested(base, feat, grid)
        s0, s1 = scores(rows, base, n_ref, n_cases), scores(rows, acc, n_ref, n_cases)
        delta, ci = paired(per_case(acc), per_case(base), rng)
        lost = s0["tp"] - s1["tp"]
        out["rules"][base_name] = dict(before=s0, after=s1, L_per_fold=chosen, delta_f1=delta, ci95=ci, tp_lost=lost, tp_lost_share=round(lost / max(s0["tp"], 1), 3))
        print(f"  rule on {base_name:22s}: F1 {s0['f1']:.3f} -> {s1['f1']:.3f} ({delta:+.3f} {ci}), P {s0['precision']:.2f}->{s1['precision']:.2f}, S {s0['sensitivity']:.2f}->{s1['sensitivity']:.2f}, FP/case {s0['fp_per_case']:.2f}->{s1['fp_per_case']:.2f}, TP lost {lost} ({100 * lost / max(s0['tp'], 1):.1f} %), L per fold {chosen}")
    if a.json:
        json.dump(out, open(absolute(a.json), "w"), indent=1)
    if a.table:
        json.dump(rows, open(absolute(a.table), "w"))


if __name__ == "__main__":
    main()
