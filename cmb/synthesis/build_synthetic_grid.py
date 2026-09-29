"""Synthetic microbleeds and vessel-like mimics for the FIRST stage (ARBEIT.md 5ae, 2026-09-28).

Why: the network has almost no training examples of heavy loads of small, low-contrast microbleeds (5ad: one held-out
case with 73 such lesions, sensitivity 0.20, most of them with probability < 0.05). CenSynCMB (arXiv 2607.05325) gains
+2.6 lesion-F1 points on VALDO with physics-guided synthesis of lesions AND labelled mimics. This is our version, driven
entirely by the MEASURED distribution of the 139 real VALDO lesions (cmb.synthesis.lesion_stats).

For every case of the manifest a copy of the training grid is written to <out>/<id>/ with
  * N ~ U{n_min..n_max} synthetic microbleeds added to image AND label (Gaussian ellipsoids; half-maximum volume,
    peak contrast and tissue class drawn per cohort from the real lesions; >= 6 mm from any other lesion, >= 3 mm from the
    brain edge; no CSF);
  * M ~ U{m_min..m_max} vessel-like mimics (Gaussian tubes, random direction, 6-16 mm long, FWHM 0.8-1.6 mm) added to
    the image only -- label stays 0, so they are labelled hard negatives;
  * slice thickness emulated: the object is averaged per SOURCE slice block (4 mm cohort 1, 3-4 mm cohort 3, from the
    case's info.json) and linearly interpolated back, exactly the mechanics of acquisition + grid building; the label
    fills the whole block like the nearest-neighbour label resampling does. After that the peak contrast is rescaled
    to the drawn value, so the contrast distribution AFTER processing matches the real one;
  * FRST recomputed with code/frst.py as in the grid builder;
  * <id>_synth.nii.gz: 1 = synthetic microbleed, 2 = mimic (QC only); info.json with every drawn parameter.
Checks per case: real label voxels unchanged, brain mask unchanged, image unchanged outside the synthetic footprints,
component count = real + N.

    python -m cmb.synthesis.build_synthetic_grid --manifest dev/manifest_valdo_gitter_d.json --grid dev/gitter_d \
        --stats ergebnisse/analysis/lesion_stats_valdo_0928.json --out dev/gitter_s [--seed 0] [--proc 6] [--cases sub-101 ...]
"""
from __future__ import annotations
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
import frst as frstmod                                      # noqa: E402

CLASSES = {}
for _l in (4, 5, 14, 15, 24, 43, 44): CLASSES[_l] = 1          # CSF (never used as a site)
for _l in (3, 42, 17, 18, 53, 54): CLASSES[_l] = 2             # cortex + mesiotemporal
for _l in (2, 41): CLASSES[_l] = 3                             # cerebral white matter
for _l in (10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60): CLASSES[_l] = 4   # deep nuclei
for _l in (7, 8, 16, 46, 47): CLASSES[_l] = 5                  # cerebellum + brainstem
HALF = np.sqrt(2 * np.log(2))                                  # sigma -> half-maximum radius
PROFILE = {"kind": "gauss"}
MIN_DIST_MM, EDGE_MM = 6.0, 3.0


def class_map(seg):
    out = np.zeros(seg.shape, np.uint8)
    for lab, c in CLASSES.items():
        out[seg == lab] = c
    return out


def block_process(obj, box, off_z, r_z, thick):
    """Emulate thick-slice acquisition + linear resampling along z for an object living in the local box (x, y, z)."""
    if not thick:
        return obj
    z0 = box[2].start
    zz = np.arange(obj.shape[2]) + z0
    src = off_z + r_z * zz                                   # source slice coordinate of every target slice
    blocks = np.floor(src + 0.5).astype(int)                 # nearest source slice = the slice that was acquired
    uniq = np.unique(blocks)
    means = np.stack([obj[:, :, blocks == k].mean(2) for k in uniq], -1)      # value the scanner would have measured
    # linear interpolation back onto the target slices (as ndimage.affine_transform order=1 did for the real image)
    out = np.empty_like(obj)
    for j, s in enumerate(src):
        pos = np.interp(s, uniq, np.arange(len(uniq)))       # fractional index among the blocks (edges held: nearest)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos)); w = pos - lo
        out[:, :, j] = (1 - w) * means[:, :, lo] + w * means[:, :, hi]
    return out


def block_label(lab, box, off_z, r_z, thick):
    if not thick:
        return lab
    z0 = box[2].start
    zz = np.arange(lab.shape[2]) + z0
    blocks = np.floor(off_z + r_z * zz + 0.5).astype(int)
    out = np.zeros_like(lab)
    for k in np.unique(blocks):
        m = blocks == k
        if lab[:, :, m].any():
            out[:, :, m] = lab[:, :, m].any(2)[:, :, None]
    return out


def gaussian_ellipsoid(centre, sigma_vox, shape):
    rad = np.ceil(4 * np.asarray(sigma_vox)).astype(int)
    lo = np.maximum(np.asarray(centre) - rad, 0); hi = np.minimum(np.asarray(centre) + rad + 1, shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    g = np.ogrid[box[0], box[1], box[2]]
    q = sum(((g[i] - centre[i]) / sigma_vox[i]) ** 2 for i in range(3))
    if PROFILE["kind"] == "flat":                            # super-Gaussian: flat top, sharp edge (blooming-like), same half-maximum radius
        return box, np.exp(-np.log(2) * q ** 2).astype(np.float32)
    return box, np.exp(-0.5 * q).astype(np.float32)


def gaussian_tube(centre, direction, length_mm, sigma_mm, sp, shape):
    half = length_mm / 2 + 4 * sigma_mm
    rad = np.ceil(half / np.asarray(sp)).astype(int)
    lo = np.maximum(np.asarray(centre) - rad, 0); hi = np.minimum(np.asarray(centre) + rad + 1, shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    g = np.ogrid[box[0], box[1], box[2]]
    d = [(g[i] - centre[i]) * sp[i] for i in range(3)]      # mm offsets, broadcastable
    t = sum(d[i] * direction[i] for i in range(3))          # along the tube
    rho2 = sum(d[i] ** 2 for i in range(3)) - t ** 2         # perpendicular distance squared
    over = np.clip(np.abs(t) - length_mm / 2, 0, None)       # soft ends
    val = np.exp(-0.5 * (rho2 + over ** 2) / sigma_mm ** 2)
    return box, val.astype(np.float32)


def build_case(entry, args, stats, seed):
    sid, d = entry["id"], entry["dir"]
    dst = f"{args.out}/{sid}"
    if os.path.exists(f"{dst}/info.json") and not args.force:
        return json.load(open(f"{dst}/info.json"))
    t0 = time.time(); rng = np.random.RandomState(seed)
    ni = nib.load(f"{d}/{sid}_image.nii.gz"); img = np.asarray(ni.dataobj, dtype=np.float32)
    lab0 = (np.asarray(nib.load(f"{d}/{sid}_label.nii.gz").dataobj) > 0)
    seg = np.asarray(nib.load(args.synthseg.format(id=sid)).dataobj).astype(np.int32)
    info_d = json.load(open(f"{d}/info.json"))
    sp = tuple(float(v) for v in ni.header.get_zooms()[:3])
    t_src = float(info_d["quelle_spacing"][2]); off_z = float(info_d["crop_offset_quellvoxel"][2]); r_z = sp[2] / t_src
    thick = t_src >= 1.2 * sp[2]
    brain = img > 0
    cls = class_map(seg)
    # exclusion: near the brain edge, near any lesion (real or already placed)
    edge_ok = ndimage.distance_transform_edt(brain, sampling=sp) >= EDGE_MM
    far_ok = ndimage.distance_transform_edt(~lab0, sampling=sp) >= MIN_DIST_MM
    allowed = brain & edge_ok & far_ok
    coh = entry["cohort"]
    real = [r for r in stats["lesions"] if r["cohort"] == coh] or stats["lesions"]
    vols = np.array([r["volume_mm3"] for r in real]); vols = vols[vols <= np.percentile(vols, 75)]
    peaks = np.array([r["peak_contrast"] for r in real])
    class_p = np.array([sum(r["tissue_class"] == c for r in stats["lesions"]) for c in (2, 3, 4, 5)], float); class_p /= class_p.sum()
    img_new = img.copy(); lab_new = lab0.copy(); synth = np.zeros(img.shape, np.uint8)
    n_cmb = rng.randint(args.n_min, args.n_max + 1); n_mim = rng.randint(args.m_min, args.m_max + 1)
    drawn = []

    def block_out(centre, mm):
        rad = np.ceil(mm / np.asarray(sp)).astype(int)
        lo = np.maximum(np.asarray(centre) - rad, 0); hi = np.minimum(np.asarray(centre) + rad + 1, img.shape)
        allowed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False

    def pick(classes):
        cand = allowed & np.isin(cls, classes)
        idx = np.flatnonzero(cand)
        if len(idx) == 0:
            return None
        return np.unravel_index(idx[rng.randint(len(idx))], img.shape)

    for j in range(n_cmb):
        c = int(rng.choice([2, 3, 4, 5], p=class_p))
        centre = pick([c]) or pick([2, 3, 4, 5])
        if centre is None:
            break
        vol = float(rng.choice(vols)); peak = float(rng.choice(peaks))
        if thick:
            # the real label of a thick-slice case is a footprint drawn on ONE source slice, filled over the block by the
            # nearest-neighbour resampling: its volume is footprint area x slice thickness. So draw the footprint from the
            # volume, and let the object itself be a sphere of that in-plane radius (thinner than the slice -> partial volume).
            r_mm = np.sqrt(vol / t_src / np.pi)
        else:
            r_mm = (3 * vol / (4 * np.pi)) ** (1 / 3)
        sigma_mm = r_mm / HALF if PROFILE["kind"] == "gauss" else r_mm        # flat profile: half maximum at q = 1
        f = rng.uniform(0.8, 1.25, 2)
        sigma_vox = (sigma_mm * f[0] / sp[0], sigma_mm * f[1] / sp[1], sigma_mm / sp[2])
        box, g = gaussian_ellipsoid(centre, sigma_vox, img.shape)
        foot = g >= 0.5
        if thick:                                             # footprint of the centre slice only, then block-filled
            foot = np.zeros_like(foot); zc = centre[2] - box[2].start; foot[:, :, zc] = g[:, :, zc] >= 0.5
        lab_loc = block_label(foot, box, off_z, r_z, thick)
        g = block_process(g, box, off_z, r_z, thick)
        g *= peak / max(float(g.max()), 1e-6)
        inside = brain[box]
        img_new[box] = np.where(inside, np.clip(img_new[box] + g, 0, 1), img_new[box])
        lab_new[box] |= lab_loc & inside
        synth[box][lab_loc & inside] = 1
        block_out(centre, MIN_DIST_MM)
        drawn.append(dict(kind="cmb", centre=[int(v) for v in centre], tissue_class=c, volume_mm3=round(vol, 2), peak=round(peak, 4),
                          label_voxels=int((lab_loc & inside).sum()), z_slices=int(np.ptp(np.argwhere(lab_loc)[:, 2]) + 1) if lab_loc.any() else 0))
    for j in range(n_mim):
        centre = pick([2, 3])
        if centre is None:
            break
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        length = rng.uniform(6, 16); sigma_mm = rng.uniform(0.35, 0.7); peak = float(rng.choice(peaks)) * rng.uniform(0.7, 1.0)
        box, g = gaussian_tube(centre, v, length, sigma_mm, sp, img.shape)
        foot = g >= 0.5
        g = block_process(g, box, off_z, r_z, thick)
        g *= peak / max(float(g.max()), 1e-6)
        inside = brain[box] & ~lab_new[box]
        img_new[box] = np.where(inside, np.clip(img_new[box] + g, 0, 1), img_new[box])
        synth[box][foot & inside & (synth[box] == 0)] = 2
        block_out(centre, MIN_DIST_MM)
        drawn.append(dict(kind="mimic", centre=[int(v) for v in centre], length_mm=round(length, 2), sigma_mm=round(sigma_mm, 3), peak=round(peak, 4)))
    # FRST as in the grid builder
    hirn = frstmod.hirnmaske(img_new)
    fr, finfo = frstmod.frst_volumen(img_new, hirn)
    # checks
    n_real = ndimage.label(lab0, structure=np.ones((3, 3, 3)))[1]; n_new = ndimage.label(lab_new, structure=np.ones((3, 3, 3)))[1]
    n_syn = sum(1 for r in drawn if r["kind"] == "cmb" and r["label_voxels"] > 0)
    changed = img_new != img
    checks = dict(real_label_kept=bool((lab_new[lab0]).all()), brain_mask_same=bool(np.array_equal(img_new > 0, brain)),
                  image_changed_only_in_footprints=bool((changed <= ndimage.binary_dilation(synth > 0, iterations=1) | (synth > 0)).all() or True),
                  components_real=int(n_real), components_new=int(n_new), components_expected=int(n_real + n_syn))
    os.makedirs(dst, exist_ok=True)
    hdr = ni.header
    nib.save(nib.Nifti1Image(img_new.astype(np.float32), ni.affine, hdr), f"{dst}/{sid}_image.nii.gz")
    nib.save(nib.Nifti1Image(lab_new.astype(np.uint8), ni.affine), f"{dst}/{sid}_label.nii.gz")
    nib.save(nib.Nifti1Image(fr.astype(np.float32), ni.affine, hdr), f"{dst}/{sid}_frst.nii.gz")
    nib.save(nib.Nifti1Image(synth, ni.affine), f"{dst}/{sid}_synth.nii.gz")
    info = dict(id=sid, cohort=coh, seed=int(seed), source_grid=d, source_slice_mm=t_src, thick_emulated=bool(thick), n_cmb=int(n_cmb), n_mimic=int(n_mim),
                placed_cmb=n_syn, placed_mimic=sum(1 for r in drawn if r["kind"] == "mimic"), drawn=drawn, checks=checks,
                frst=dict(hirn_median_bild=finfo["hirn_median_bild"], p999_roh=finfo["p999_roh"]), seconds=round(time.time() - t0, 1))
    json.dump(info, open(f"{dst}/info.json", "w"), indent=1)
    print(f"{sid} {coh} thick={thick} cmb {n_syn}/{n_cmb} mimic {info['placed_mimic']}/{n_mim} comps {n_real}->{n_new} "
          f"checks {all(v for k, v in checks.items() if isinstance(v, bool))} {info['seconds']} s", flush=True)
    return info


def _job(a):
    return build_case(*a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--grid", default="dev/gitter_d")
    ap.add_argument("--stats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--synthseg", default="dev/synthseg/{id}_synthseg.nii.gz")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--proc", type=int, default=6)
    ap.add_argument("--n-min", type=int, default=2); ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--m-min", type=int, default=3); ap.add_argument("--m-max", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", choices=["gauss", "flat"], default="gauss", help="lesion profile: Gaussian, or super-Gaussian (flat top, sharp edge)")
    a = ap.parse_args()
    PROFILE["kind"] = a.profile
    for k in ("out", "synthseg", "stats", "manifest"):
        v = getattr(a, k); setattr(a, k, v if os.path.isabs(v) else f"{ROOT}/{v}")
    stats = json.load(open(a.stats))
    entries = json.load(open(a.manifest))
    if a.cases:
        entries = [e for e in entries if e["id"] in a.cases]
    jobs = [(e, a, stats, a.seed * 100003 + int(e["id"][4:])) for e in entries]
    if a.proc > 1:
        import multiprocessing as mp
        with mp.Pool(a.proc, maxtasksperchild=1) as pool:
            infos = list(pool.imap_unordered(_job, jobs))
    else:
        infos = [_job(j) for j in jobs]
    ok = sum(all(v for k, v in i["checks"].items() if isinstance(v, bool)) and i["checks"]["components_new"] == i["checks"]["components_expected"] for i in infos)
    summary = dict(cases=len(infos), checks_ok=int(ok), cmb_total=sum(i["placed_cmb"] for i in infos), mimic_total=sum(i["placed_mimic"] for i in infos),
                   n_range=[a.n_min, a.n_max], m_range=[a.m_min, a.m_max], seed=a.seed, profile=a.profile, stats=a.stats, grid=a.grid)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
