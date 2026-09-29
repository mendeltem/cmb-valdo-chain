#!/usr/bin/env python3
"""we5: post-processing (26-connectivity filter in mm3 + FP filter via T1 SynthSeg) and ONE-TIME
validation measurement on the validation set from we4 (VALDO Task 2, T2*).

Hard rules (task + sign-off 11 Sep):
  - All filter parameters are fixed BEFOREHAND on the CV folds (57 cases) and
    frozen into dev/filterkette.json. The validation set (15 cases) is touched
    exactly ONCE after that. Whoever still tunes the filter afterward has burned the set.
  - Size filter in mm3 (voxel x pixdim, per case) -- a voxel threshold would be
    cohort-dependent (cohorts have very different voxel volumes).
  - FP filter via T1: mri_synthseg on sub-XXX_space-T2S_desc-masked_T1.nii.gz.
    NO registration -- the T1 is in T2S space (72/72 checked, we1 contract).
    BUT: the T1 is on the ORIGINAL T2S grid, NOT in the Grid-C space of the
    predictions (0.5x0.5x1.0 mm, cropped). The SynthSeg label is therefore
    converted to the Grid-C raster with nearest neighbour (no interpolation, label
    stays discrete) -- a grid conversion, not a registration.
    Two variants, measured separately, each with TP loss:
      V1 ventricle:  labels 4,12,13,22,23 (lat./infer. lat. ventricle l/r, 3rd, 4th)
      V2 +CSF:       V1 + label 17 (free CSF)
    (label table from mri_synthseg --vol-CSV, 33 structures, checked on sub-201;
    wd6 comparison value: SWI cohort, there the CSF variant cost 12 TP -- measure fresh here.)
  - Stages are only kept if they improve F1 on the CV (pooled, 57 cases).
  - Validation: ONCE, best model from we4 = config r3, 5 fold models, each with its
    out-of-fold threshold and out-of-fold minimum voxel size; majority vote (>=3 of 5).
  - Reference = gitter_c label (26-connectivity components), measured exactly like we4 (label_for).
  - Numbers as mean +- std, never median. Measurement and interpretation kept separate.

Call:
  python we5.py --probe             # 2 cases: 1 CV + 1 val through ALL stages, time it
  python we5.py --synthseg [ids]    # mri_synthseg (CPU, parallel) + resample onto Grid C
  python we5.py --cv                # filter tuning on 57 CV predictions ->
                                    # dev/filterkette.json (FREEZE)
  python we5.py --validierung       # ONE-TIME: 15 val cases -> ergebnisse/validierung/
                                    # summary.json (with filterkette_sha256)

Files:
  dev/synthseg/<id>_synthseg.nii.gz           SynthSeg in Grid-C space (resampled)
  dev/filterkette.json                        frozen chain (before the validation!)
  ergebnisse/validierung/<id>_pred.nii.gz     prediction per val case (exactly one)
  ergebnisse/validierung/summary.json         validation numbers + CV alongside
"""
import os, sys, json, time, hashlib, argparse, subprocess, glob, shutil
import numpy as np
import nibabel as nib
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
EXTERN = "/home/uchralt/data/extern/valdo2021/Task2"
N26 = np.ones((3, 3, 3), dtype=bool)
FS = "/home/uchralt/freesurfer/8.2.0/bin/mri_synthseg"
V1_VENTRIKEL = {4, 12, 13, 22, 23}          # lat./infer. lat. ventricle (l/r), 3rd, 4th ventricle
V2_CSF = V1_VENTRIKEL | {17}                 # + free CSF
MAX_KOMP_VOX = 2100                            # blob guard from we4 (in voxels)
MM3_GRID = (1, 2, 3, 5, 8, 12, 20, 30, 50)    # size filter in mm3 (per case: voxel x pixdim)
MM3_TOP = (2, 5, 8)                            # only these 3 points get the aspect grid
ASPEKT_GRID = (1.25, 1.5, 2.0, 3.0)            # shape filter: max/min bbox half-length

sys.path.insert(0, f"{ROOT}/code")
from metric import hits, summarize
from metric_valdo import label_for

# ------------------------------------------------------------------- basic filter operations

def komponenten(bin_map):
    """26-connectivity components as a list of (voxel count, bbox half-lengths)."""
    lab, n = ndimage.label(bin_map, structure=N26)
    out = []
    for c in range(1, n + 1):
        m = lab == c
        z = np.argwhere(m)
        out.append((int(m.sum()), (z.max(0) - z.min(0) + 1) / 2.0))
    return lab, out

def aspekt(bb_halb):
    bb = np.maximum(bb_halb, 1.0)
    return float(bb.max() / bb.min())

def filter_mind_mm3(bin_map, vox_mm3, mind_mm3, max_komp_vox=MAX_KOMP_VOX, aspekt_max=None):
    """Remove 26-connectivity components: < mind_mm3, > max_komp_vox voxels, or aspect > aspekt_max."""
    lab, comps = komponenten(bin_map)
    out = np.zeros_like(bin_map)
    for c in range(1, lab.max() + 1):
        vox, bb = comps[c - 1]
        if vox > max_komp_vox:
            continue
        if vox * vox_mm3 < mind_mm3:
            continue
        if aspekt_max is not None and aspekt(bb) > aspekt_max:
            continue
        out |= (lab == c)
    return out

def fp_filter_synthseg(pred_bin, vent_mask):
    """Discard candidate components whose CENTROID falls inside the ventricle/CSF mask.
    (Task: 'whose centroid falls in ventricle/CSF'; not a touch filter.)"""
    lab, n = ndimage.label(pred_bin, structure=N26)
    out = np.zeros_like(pred_bin)
    for c in range(1, n + 1):
        cen = ndimage.center_of_mass(lab == c)
        if vent_mask[tuple(np.round(cen).astype(int))]:
            continue
        out |= (lab == c)
    return out

def resample_labels(lab, src_aff, gc_aff, gc_shape):
    """Labels in the source raster (src_aff) onto the Grid-C raster (nearest neighbour,
    cval=0 = background). Grid conversion via world coordinates, not a registration."""
    grids = np.meshgrid(*[np.arange(s, dtype=float) for s in gc_shape], indexing="ij")
    xyz = np.stack([g.ravel() for g in grids], 0)
    world = gc_aff @ np.vstack([xyz, np.ones(len(xyz[0]))[None, :]])
    src_vox = np.linalg.inv(src_aff) @ world
    # src_vox is (4, N) in homogeneous coordinates; resample only the first 3 axes.
    return ndimage.map_coordinates(lab, src_vox[:3].reshape(3, *gc_shape),
                                   order=0, mode="constant", cval=0).astype(np.uint8)

def synthseg_maske(pid, variant):
    """Resample SynthSeg (T1 grid) onto Grid C (nearest neighbour), boolean mask."""
    seg = nib.load(f"{ROOT}/dev/synthseg/{pid}_synthseg.nii.gz")
    gc = nib.load(f"{ROOT}/dev/gitter_c/{pid}/{pid}_image.nii.gz")
    lab = np.asarray(seg.dataobj)
    if lab.shape != gc.shape:
        raise SystemExit(f"{pid}: synthseg {lab.shape} != grid C {gc.shape}")
    return np.isin(lab, list(variant))

def kette_anwenden(pred_bin, vox_mm3, kette, vent_mask=None):
    """Apply the frozen chain to a binary prediction (Grid-C space)."""
    out = pred_bin
    if kette.get("groesse_mm3"):
        out = filter_mind_mm3(out, vox_mm3, kette["groesse_mm3"],
                              aspekt_max=kette.get("aspekt"))
    if kette.get("synthseg_variant") and vent_mask is not None:
        out = fp_filter_synthseg(out, vent_mask)
    return out

# ------------------------------------------------------------------- SynthSeg call

def synthseg_lauf(ids, parallel=3):
    """mri_synthseg (CPU, 8 threads per case) + resample onto Grid C."""
    os.makedirs(f"{ROOT}/dev/synthseg", exist_ok=True)
    zu_tun = [i for i in ids if not os.path.exists(f"{ROOT}/dev/synthseg/{i}_synthseg.nii.gz")]
    if not zu_tun:
        print("all SynthSeg already done"); return
    def ein(id_):
        t1 = f"{EXTERN}/{id_}/{id_}_space-T2S_desc-masked_T1.nii.gz"
        o = f"{ROOT}/dev/synthseg/{id_}_roh"
        t0 = time.time()
        # NaN repair (coach 14 Sep 21:5x): 33/72 delivered T1s carry NaN as an
        # edge shell outside the brain volume (triple-eroded brain mask NaN-free,
        # sub-304: one connected component at the volume edge, sub-102: two).
        # NaN poisons the segmenter's percentile normalisation -> empty
        # segmentation, rc 0, the case disappears silently. Zeroing does not invent
        # anatomy. Cleaned TEMPORARY copy; the delivered file stays untouched.
        im = nib.load(t1)
        a = np.asanyarray(im.dataobj)
        nan0 = t1 + ".nan0.nii.gz"
        if np.isnan(a).any():
            nib.save(nib.Nifti1Image(np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0),
                                     im.affine), nan0)
            t1 = nan0
        r = subprocess.run([FS, "--i", t1, "--o", o, "--cpu", "--threads", "8", "--noaddctab"],
                           capture_output=True, text=True)
        if os.path.exists(nan0):
            os.remove(nan0)
        if r.returncode != 0:
            return id_, f"rc={r.returncode}: {r.stderr[:200]}"
        f = glob.glob(f"{o}/*_synthseg.nii.gz")
        if not f:
            return id_, "no output"
        # newest file (mtime), not alphabetical: a stale raw output from an
        # earlier run (e.g. before the NaN repair) must not shadow the current one
        f = [max(f, key=os.path.getmtime)]
        roh = nib.load(f[0])
        gc = nib.load(f"{ROOT}/dev/gitter_c/{id_}/{id_}_image.nii.gz")
        zi = resample_labels(np.asarray(roh.dataobj), roh.affine, gc.affine, gc.shape)
        u = np.unique(zi)
        if len(u) < 20:
            return id_, f"SUSP: only {len(u)} labels after resample"
        nib.save(nib.Nifti1Image(zi, gc.affine), f"{ROOT}/dev/synthseg/{id_}_synthseg.nii.gz")
        # remove raw output after success (rmtree: mri_synthseg leaves more
        # than one file in the directory, os.rmdir fails with ENOTEMPTY)
        shutil.rmtree(o, ignore_errors=True)
        return id_, f"ok {time.time()-t0:.0f}s labels={len(u)}"
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for id_, st in ex.map(ein, zu_tun):
            print(f"synthseg {id_}: {st}", flush=True)

# ------------------------------------------------------------------- CV: predictions + measurement

def cv_faelle():
    """57 CV cases: (id, bin_pred, vox_mm3, ref) on Grid C."""
    d = json.load(open(f"{ROOT}/dev/split.json"))
    out = []
    for k, fids in enumerate(d["folds"]):
        for pid in fids:
            pp = f"{ROOT}/ergebnisse/cv5-best/fold{k}/{pid}_pred.nii.gz"
            if not os.path.exists(pp):
                raise SystemExit(f"CV prediction missing: {pp}")
            pi = nib.load(pp)
            ref = np.asarray(label_for(pid, pi).dataobj) > 0
            out.append((pid, np.asarray(pi.dataobj) > 0,
                        float(abs(np.linalg.det(pi.affine[:3, :3]))), ref))
    assert len(out) == 57, len(out)
    return out

def messen(faelle, kette=None, ventmasken=None):
    """Pooled 26-connectivity measurement; ventmasken: {pid: {variant: boolarray}} or None."""
    rows = []
    for pid, pm, vox_mm3, ref in faelle:
        out = pm
        if kette:
            vm = ventmasken[pid][kette["synthseg_variant"]] if kette.get("synthseg_variant") else None
            out = kette_anwenden(pm, vox_mm3, kette, vent_mask=vm)
        rows.append(dict(id=pid, **hits(ref, out)))
    return summarize(rows), rows

def cv_abstimmung(faelle, ventmasken):
    """Measure all stages on the CV, best chain (only F1-improving stages)."""
    roh, _ = messen(faelle)
    print(f"CV raw: F1 {roh['f1']} sens {roh['sensitivity']} fp/case {roh['fp_per_case']} "
          f"(n_pred {roh['n_pred']})", flush=True)

    beste = (roh["f1"], dict(groesse_mm3=None, aspekt=None, synthseg_variant=None))
    grid_z = {}
    for mm3 in MM3_GRID:
        asps = ASPEKT_GRID if mm3 in MM3_TOP else [None]
        for asp in asps:
            s, _ = messen(faelle, dict(groesse_mm3=mm3, aspekt=asp, synthseg_variant=None))
            grid_z[f"{mm3}mm3_a{asp}"] = s["f1"]
            if s["f1"] > beste[0]:
                beste = (s["f1"], dict(groesse_mm3=mm3, aspekt=asp, synthseg_variant=None))
    b_f1, b_k = beste
    print(f"CV best size stage: F1 {b_f1} chain {b_k}", flush=True)

    # SynthSeg stage: on the best size stage, V1 and V2 separately
    stufen = {"roh": roh}
    stufen["groesse"] = messen(faelle, dict(groesse_mm3=b_k["groesse_mm3"],
                                            aspekt=b_k["aspekt"], synthseg_variant=None))[0] \
        if b_k["groesse_mm3"] else roh
    for v in ("V1", "V2"):
        k = dict(b_k, synthseg_variant=v)
        s, _ = messen(faelle, k, ventmasken)
        stufen[v] = s
        print(f"CV {v}: F1 {s['f1']} sens {s['sensitivity']} fp/case {s['fp_per_case']} "
              f"tp {s['tp']} (baseline tp {stufen['groesse']['tp']})", flush=True)
        if s["f1"] > beste[0]:
            beste = (s["f1"], k)

    kette = beste[1]
    # report TP losses per SynthSeg variant (against the size stage)
    tp_verlust = {v: stufen["groesse"]["tp"] - stufen[v]["tp"] for v in ("V1", "V2")}
    print(f"FROZEN: F1 {beste[0]} chain {kette} TP loss {tp_verlust}", flush=True)

    return dict(
        kette=kette,
        cv_roh=roh,
        cv_groesse=stufen["groesse"],
        cv_v1=stufen["V1"],
        cv_v2=stufen["V2"],
        tp_verlust_v1=tp_verlust["V1"],
        tp_verlust_v2=tp_verlust["V2"],
        groesse_grid_f1=grid_z,
        entscheidung=("keep a stage only if it improves F1 on the CV (pooled, 57 cases); "
                      "size filter in mm3 per case (voxel x pixdim); "
                      "SynthSeg filter: centroid of the candidate component in "
                      "ventricles (V1) / ventricles+CSF (V2); V1 and V2 measured separately."),
        eingefroren_am=time.strftime("%Y-%m-%d %H:%M:%S"),
        hinweis=("chain tuned on the CV folds and FROZEN before the validation set "
                 "(15 cases) was touched. Any further adjustment after looking at the "
                 "validation numbers is forbidden (the set would be burnt)."),
    )

# ------------------------------------------------------------------- Validation (ONCE)

def validierung_einmal(fk_pf, probe=False, schwelle=None):
    import torch
    import cv5
    cv5.DEV = os.environ.get("WE5_DEV", "cuda")
    fk = json.load(open(fk_pf))
    kette = fk["kette"]
    val_ids = json.load(open(f"{ROOT}/dev/split.json"))["validation"]
    if probe:
        val_ids = val_ids[:2]
        outd = f"{ROOT}/dev/we5_probe"
    else:
        outd = f"{ROOT}/ergebnisse/validierung"
    os.makedirs(outd, exist_ok=True)

    # 5 r3 fold models (best model from we4) with out-of-fold parameters
    mods = []
    for k in range(5):
        meta = json.load(open(f"{ROOT}/dev/modelle/best/fold{k}/meta.json"))
        assert meta["konf"] == "r3"
        m, nch = cv5.neues_modell(meta["params"])
        m.load_state_dict(torch.load(f"{ROOT}/dev/modelle/best/fold{k}/modell.pt", map_location="cpu"))
        m.to(cv5.DEV); m.eval()
        mods.append((m, meta["schwelle"], meta["mindestgroesse"]))
    print(f"{len(mods)} r3 models loaded (thresholds "
          f"{[m[1] for m in mods]}, mg voxels {[m[2] for m in mods]})", flush=True)

    cv5.lade_daten(ids=None)   # fold cases only; validation is not loaded (fold -1)
    # bring val cases into the recs structure (same Grid-C files)
    d = json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]
    for c in d:
        if c["fold"] != -1:
            continue
        sid = c["id"]
        ni = nib.load(f"{cv5.GITTER}/{sid}/{sid}_image.nii.gz")
        dreh = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
        cv5.FA["recs"][sid] = dict(
            img=dreh(np.asarray(ni.dataobj, dtype=np.float32)),
            frs=dreh(np.asarray(nib.load(f"{cv5.GITTER}/{sid}/{sid}_frst.nii.gz").dataobj, dtype=np.float32)),
            lab=dreh((np.asarray(nib.load(f"{cv5.GITTER}/{sid}/{sid}_label.nii.gz").dataobj) > 0).astype(np.uint8)),
            komp=[], fold=-1, affine=ni.affine)

    # SynthSeg masks (V1/V2) in Grid-C space
    ventmasken = {pid: {"V1": synthseg_maske(pid, V1_VENTRIKEL), "V2": synthseg_maske(pid, V2_CSF)}
                  for pid in val_ids}
    # precision check of the resample mask
    for pid in val_ids[:1]:
        print(f"{pid} synthseg mask: V1 {int(ventmasken[pid]['V1'].sum())} voxels, "
              f"V2 {int(ventmasken[pid]['V2'].sum())} voxels", flush=True)

    rows = []
    t0 = time.time()
    for i, pid in enumerate(val_ids):
        gc = nib.load(f"{ROOT}/dev/gitter_c/{pid}/{pid}_image.nii.gz")
        vox_mm3 = float(abs(np.linalg.det(gc.affine[:3, :3])))
        votes = np.zeros_like(np.asarray(gc.dataobj), dtype=np.uint8)
        for m, thr, mg_vox in mods:
            p, nk = cv5.proba_fall(m, pid)
            # proba_fall returns cv5's internal orientation (x,y,z) (recs via dreh=transpose(2,1,0));
            # votes/ventmasken are in the native Grid-C space (z,y,x) -> rotate back like cv5.py line 340/347
            p = np.ascontiguousarray(p.transpose(2, 1, 0))
            b = p > (schwelle if schwelle is not None else thr)
            # out-of-fold minimum voxel size (we4: in voxels)
            lab, n = ndimage.label(b, structure=N26)
            b2 = np.zeros_like(b)
            for c in range(1, n + 1):
                if mg_vox <= (lab == c).sum() <= MAX_KOMP_VOX:
                    b2 |= (lab == c)
            votes += b2
        pred_bin = votes >= 3
        pred_fin = kette_anwenden(pred_bin, vox_mm3, kette, vent_mask=ventmasken[pid][kette["synthseg_variant"]] if kette.get("synthseg_variant") else None)
        nib.save(nib.Nifti1Image(pred_fin.astype(np.uint8), gc.affine), f"{outd}/{pid}_pred.nii.gz")
        ref = np.asarray(label_for(pid, nib.load(f"{outd}/{pid}_pred.nii.gz")).dataobj) > 0
        t = hits(ref, pred_fin)
        rows.append(dict(id=pid, **t))
        print(f"[{i+1}/{len(val_ids)}] {pid}: ref={t['n_ref']} pred={t['n_pred']} "
              f"tp={t['tp']} fp={t['fp']} fn={t['fn']}", flush=True)
        torch.cuda.empty_cache()

    z = summarize(rows)
    z["cases"] = len(val_ids)
    z["n_pred"] = int(sum(r["n_pred"] for r in rows))
    z["filterkette_sha256"] = hashlib.sha256(open(fk_pf, "rb").read()).hexdigest()
    z["modell"] = ("we4 winner r3: 5 fold models, per-fold out-of-fold minimum voxel size, "
                   "majority vote >=3/5"
                   + (f", threshold {schwelle}" if schwelle is not None
                      else ", per-fold out-of-fold threshold"))
    z["filterkette"] = kette
    z["cv_daneben"] = dict(roh=fk["cv_roh"], groesse=fk["cv_groesse"],
                           v1=fk["cv_v1"], v2=fk["cv_v2"])
    z["unsicherheit"] = ("47 reference microbleeds carry about +-0.1 F1 uncertainty; "
                         "a strong deviation from the CV is a finding, not a call "
                         "for re-tuning (the set was touched exactly once).")
    z["dauer_s"] = round(time.time() - t0, 1)
    z["probe"] = probe
    tmp = f"{outd}/summary.json.tmp"
    json.dump(z, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, f"{outd}/summary.json")
    print("VALIDATION DONE ->", f"{outd}/summary.json", flush=True)
    print(json.dumps(z, indent=1, ensure_ascii=False), flush=True)
    return z

# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--validierung", action="store_true")
    ap.add_argument("--synthseg", nargs="*", default=None)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--schwelle", type=float, default=None,
                    help="applies to --validierung: global threshold instead of per-fold out-of-fold threshold")
    a = ap.parse_args()
    if a.synthseg is not None:
        synthseg_lauf(a.synthseg, a.parallel)
    elif a.cv:
        faelle = cv_faelle()
        print(f"{len(faelle)} CV cases loaded", flush=True)
        ventmasken = {pid: {"V1": synthseg_maske(pid, V1_VENTRIKEL),
                            "V2": synthseg_maske(pid, V2_CSF)} for pid, *_ in faelle}
        fk = cv_abstimmung(faelle, ventmasken)
        tmp = f"{ROOT}/dev/filterkette.json.tmp"
        json.dump(fk, open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, f"{ROOT}/dev/filterkette.json")
        print("dev/filterkette.json FROZEN", flush=True)
    elif a.probe:
        # 1 CV case (filter paths) + 2 val cases (inference time), with timing
        faelle = cv_faelle()[:1]
        pid = faelle[0][0]
        v = {"V1": synthseg_maske(pid, V1_VENTRIKEL), "V2": synthseg_maske(pid, V2_CSF)}
        t0 = time.time()
        roh, _ = messen(faelle)
        t1 = time.time()
        s, _ = messen(faelle, dict(groesse_mm3=5, aspekt=None, synthseg_variant=None), {"V1": v, "V2": v})
        s, _ = messen(faelle, dict(groesse_mm3=5, aspekt=None, synthseg_variant="V2"), {"V1": v, "V2": v})
        t2 = time.time()
        print(f"PROBE CV case {pid}: raw F1 {roh['f1']}, filter {t1-t0:.2f}s, +SynthSeg {t2-t1:.2f}s")
        validierung_einmal(f"{ROOT}/dev/filterkette.json", probe=True)
        print("PROBE OK", flush=True)
    elif a.validierung:
        validierung_einmal(f"{ROOT}/dev/filterkette.json", schwelle=a.schwelle)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
