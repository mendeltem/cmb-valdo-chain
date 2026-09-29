#!/usr/bin/env python3
"""we4 CV5, rebuild 11 Sep 2026 (Claude, after three code reviews): 5-fold CV of the 3D attention U-Net
on VALDO Task 2 (T2*), with one inner fold for epoch AND threshold.

Why rebuilt (findings 11 Sep, board [521]-[530], backup dev/cv5.py.vor-umbau-0911):
  - Every lesion sat exactly in the patch center -> the network learned 'lesion = center' and stayed
    silent on the tiled full maps (code/archiv/versatz_test.py: peak 1.0 centered, 0.0 from 10 voxels on).
  - val_dice measured centered TRAINING patches; the threshold was chosen on the training cases
    -> model and threshold choice on the training fit.
  - filtert_m kept label 0 (whole image area as prediction, F1 ~0.95 faked).
  - Patch axes swapped: arrays are (x,y,z), PATCH was meant as (z,y,x).
  - FRST > 0.5 was almost empty; now a continuous channel from code/frst.py (dev/gitter_c).
  - the we3 recipe (120 patches/epoch, focal Tversky) did not learn; the we2/SWI recipe learns.

Procedure per configuration and test fold k (none of this sees the test fold before the prediction):
  inner fold i = (k+1) % 5, training on the three remaining folds.
  Every EVAL_JEDE epochs: full maps of the inner fold -> 26-connectivity F1 over the grid
  (threshold x min size, clump guard) -> best state (epoch, threshold, min size).
  Early stop after GEDULD evaluations without improvement (earliest from MIN_EPOCHEN on).
  Test fold k: full maps with the best state, binarized with the inner choice.
The validation set (fold -1) is never read.

Data: dev/gitter_c/<id>/<id>_{image,frst,label}.nii.gz (0.5x0.5x1.0 mm, cropped to the brain,
NIfTI axes x,y,z). When loading -> (z,y,x); PATCH (38,62,94) = 38x31x47 mm.
Predictions are rotated back to (x,y,z) and stored in grid-C space.

Usage:
  python cv5.py --selbsttest        # CPU: fast grid search == metric.hits, offset within the patch
  python cv5.py --probe             # ONE fold complete with mini settings (every path
                                    # run through once) + timing -> dev/cv5_probe.json
  python cv5.py [--konf r1 r2] [--falten 0 1] [--zeit S]
Results: ergebnisse/cv5-<konf>/fold<k>/{<id>_pred.nii.gz, <id>_pred_proba.nii.gz (uint8,
p*255), <id>_pred_meta.json, modell.pt, meta.json, verlauf.json}, dev/cv5_status.json.
Return value: 0 all ok, 1 at least one fold with an error, 3 time limit (resumable).
"""
import os, sys, json, time, shutil, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
DEV = os.environ.get("CV5_DEV", "cuda")
sys.path.insert(0, f"{ROOT}/code")
from attention_unet import AttentionUNet3D
from metric import hits, summarize

N26 = np.ones((3, 3, 3), bool)
GITTER = os.environ.get("CV5_GITTER", f"{ROOT}/dev/gitter_c")
PATCH = (38, 62, 94)          # (z, y, x) after rotating = 38 x 31 x 47 mm (gitter.json intent)
SEED = 42
RAND_VERSATZ = 4              # lesion sits at least this many voxels away from the patch edge
MAX_KOMP_VOX = 2100           # clump guard: > 10-mm sphere (524 mm3 / 0.25 mm3) is not a CMB;
LETZTER_STAND = False          # 21 Sep 2026: True = the test prediction comes from the LAST intermediate state (epoch max_ep), not the best of the inner fold.
                              # largest reference CMB 928 voxels. metric.hits counts every touched
                              # reference as TP and a touched prediction never as FP.
THRESH_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SIZE_GRID = (1, 2, 3, 5)
FALLBACK = (0.5, 1)           # if every grid cell has F1 = 0
EVAL_JEDE = 5                 # epochs between two evaluations of the inner fold
MIN_EPOCHEN = 10
MAX_EPOCHEN = 40
GEDULD = 3                    # evaluations without improvement until early stop
FOLD_TIME_MAX_S = 3 * 3600    # time fence per fold, checked before every epoch and phase

# Recipe = we2/SWI recipe (gitter_train.py; repos/Microbleed_Attention_3D_Unet), the only one that
# measurably learned on 11 Sep. Configurations deviate only in the named keys.
REZEPT = dict(n_patches=800, bs=4, lr=3e-4, weight_decay=1e-4, loss="bce_dice",
              anteil_laesion=0.6, anteil_hirn=0.25, augment=True, kanaele="t2s_frst",
              basis=32, tiefe=4, gates=True, dropout=0.0, norm="batch")
KONFS = {
    "r1": {},                                  # main recipe: T2* + continuous FRST, BatchNorm
    "r2": {"kanaele": "t2s"},                  # does the FRST contribute anything?
    "r3": {"norm": "instance"},                # BatchNorm at batch 4 vs. InstanceNorm
}

def konf_params(name):
    return {**REZEPT, **KONFS[name]}

# ---------------------------------------------------------------- Data
FA = None   # recs: sid -> dict(img, frs, lab, komp, fold, affine)

def lade_daten(ids=None):
    """Load fold cases from GITTER, rotate to (z,y,x). Validation (fold -1) never."""
    global FA
    cases = json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]
    recs = {}
    for c in cases:
        if c["fold"] == -1 or (ids is not None and c["id"] not in ids):
            continue
        sid, d = c["id"], f"{GITTER}/{c['id']}"
        ni = nib.load(f"{d}/{sid}_image.nii.gz")
        dreh = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
        img = dreh(np.asarray(ni.dataobj, dtype=np.float32))
        frs = dreh(np.asarray(nib.load(f"{d}/{sid}_frst.nii.gz").dataobj, dtype=np.float32))
        lab = dreh((np.asarray(nib.load(f"{d}/{sid}_label.nii.gz").dataobj) > 0).astype(np.uint8))
        assert img.shape == frs.shape == lab.shape, sid
        ll, n = ndimage.label(lab, structure=N26)
        komp = [np.argwhere(ll == j) for j in range(1, n + 1)]
        recs[sid] = dict(img=img, frs=frs, lab=lab, komp=komp, fold=c["fold"], affine=ni.affine)
    FA = dict(recs=recs)
    n_k = sum(len(r["komp"]) for r in recs.values())
    print(f"Data: {len(recs)} fold cases from {os.path.basename(GITTER)}, {n_k} CMB components "
          f"(validation not loaded)", flush=True)

def zufalls_start(c, shape, rng):
    """Patch start such that voxel c sits at a RANDOM position within the patch (>= RAND_VERSATZ from
    the edge, as far as the volume allows). If the volume is smaller than PATCH, start 0."""
    st = []
    for i in range(3):
        o = rng.randint(RAND_VERSATZ, max(PATCH[i] - RAND_VERSATZ, RAND_VERSATZ + 1))
        st.append(int(min(max(c[i] - o, 0), max(shape[i] - PATCH[i], 0))))
    return tuple(st)

def schnitt(a, s):
    """Patch from start s; if the volume is smaller than PATCH, it is padded with 0."""
    z, y, x = s
    p = a[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]]
    if p.shape != PATCH:
        p = np.pad(p, [(0, PATCH[i] - p.shape[i]) for i in range(3)])
    return p

def stichprobe(train_ids, rng, P):
    """Recipe sampling: anteil_laesion lesion (component uniformly distributed, so large CMBs
    don't dominate), anteil_hirn brain voxels, the rest arbitrary voxels; all with random
    offset; flip augmentation per axis (image and label the same)."""
    recs = FA["recs"]
    komps = [(s, j) for s in train_ids for j in range(len(recs[s]["komp"]))]
    n = P["n_patches"]
    n_l, n_h = int(n * P["anteil_laesion"]), int(n * P["anteil_hirn"])
    nch = 1 if P["kanaele"] == "t2s" else 2
    xs, ys = np.empty((n, nch) + PATCH, np.float32), np.empty((n, 1) + PATCH, np.float32)
    for j in range(n):
        if j < n_l and komps:
            s, k = komps[rng.randint(len(komps))]
            vox = recs[s]["komp"][k]; c = vox[rng.randint(len(vox))]
        else:
            s = train_ids[rng.randint(len(train_ids))]
            img = recs[s]["img"]
            for _ in range(50):
                c = tuple(rng.randint(d) for d in img.shape)
                if j >= n_l + n_h or img[c] != 0:
                    break
        r = recs[s]; st = zufalls_start(c, r["img"].shape, rng)
        p = np.stack([schnitt(r["img"], st)] + ([schnitt(r["frs"], st)] if nch == 2 else []), 0)
        l = schnitt(r["lab"], st)[None].astype(np.float32)
        if P["augment"]:
            for ax in (1, 2, 3):
                if rng.rand() < 0.5:
                    p, l = np.flip(p, ax), np.flip(l, ax)
        xs[j], ys[j] = p, l
    idx = rng.permutation(n)
    return torch.from_numpy(xs[idx]), torch.from_numpy(ys[idx])

# ---------------------------------------------------------------- Loss
def loss_bce_dice(lg, ziel):
    p = torch.sigmoid(lg)
    bce = F.binary_cross_entropy_with_logits(lg, ziel)
    pf, zf = p.flatten(1), ziel.flatten(1)
    dice = 1 - (2 * (pf * zf).sum(1) + 1.0) / (pf.sum(1) + zf.sum(1) + 1.0)
    return 0.5 * bce + 0.5 * dice.mean()

def loss_focal_tversky(lg, ziel, alpha=0.3, beta=0.7, gamma=4 / 3):
    """Abraham & Khan 2019: TI = TP / (TP + alpha FP + beta FN), FTL = (1 - TI)^(1/gamma).
    (The we3 version had FP 1.0 / FN 0.7 and exponent 2 -- preferred 'report nothing'.)"""
    p = torch.sigmoid(lg)
    pf, zf = p.flatten(1), ziel.flatten(1)
    tp = (pf * zf).sum(1); fp = (pf * (1 - zf)).sum(1); fn = ((1 - pf) * zf).sum(1)
    ti = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return (1 - ti).clamp_min(0).pow(1 / gamma).mean()

VERLUSTE = {"bce_dice": loss_bce_dice, "focal_tversky": loss_focal_tversky}

# ---------------------------------------------------------------- Full map
def proba_fall(model, sid, *, bs=8, nch=2):
    """Probability map (z,y,x) with half stride, overlap averaging. Tiles whose
    image is entirely 0 (outside the brain) are skipped and stay 0.
    Returns (map, number_of_tiles)."""
    r = FA["recs"][sid]
    img, frs = r["img"], r["frs"]
    shp = img.shape
    padded = tuple(max(shp[i], PATCH[i]) for i in range(3))
    starts = []
    for i in range(3):
        st = list(range(0, padded[i] - PATCH[i] + 1, PATCH[i] // 2))
        if st[-1] + PATCH[i] < padded[i]:
            st.append(padded[i] - PATCH[i])
        starts.append(st)
    pred = np.zeros(padded, np.float32); cnt = np.zeros(padded, np.float32)
    kacheln = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    kacheln = [s for s in kacheln if schnitt(img, s).any()]
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        for i in range(0, len(kacheln), bs):
            b = kacheln[i:i + bs]
            xb = np.stack([np.stack([schnitt(img, s)] + ([schnitt(frs, s)] if nch == 2 else []), 0)
                           for s in b])
            out = torch.sigmoid(model(torch.from_numpy(xb).to(DEV))).float().cpu().numpy()
            for k, (z, y, x) in enumerate(b):
                pred[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += out[k, 0]
                cnt[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += 1.0
    pred = (pred / np.maximum(cnt, 1))[:shp[0], :shp[1], :shp[2]]
    return pred, len(kacheln)

# ---------------------------------------------------------------- Post-filter + fast grid search
def filtert_m(p, thr, mind, zaehler=None):
    """p > thr, then discard components < mind or > MAX_KOMP_VOX (never the background)."""
    m = p > thr
    if not m.any():
        return m
    labl, n = ndimage.label(m, structure=N26)
    sizes = np.bincount(labl.ravel())
    keep = (sizes >= mind) & (sizes <= MAX_KOMP_VOX); keep[0] = False
    if zaehler is not None:
        gross = sizes[1:] > MAX_KOMP_VOX
        zaehler["klumpen"] = zaehler.get("klumpen", 0) + int(gross.sum())
        zaehler["klumpen_voxel"] = zaehler.get("klumpen_voxel", 0) + int(sizes[1:][gross].sum())
    return keep[labl]

def gitter_zeilen(p, lab):
    """hits() rows for ALL (threshold, min size) at once -- identical to
    metric.hits(lab, filtert_m(p, thr, mind)), but: label the reference once, label once per
    threshold, hits from the overlap table, everything within the crop around (p > smallest
    threshold) and the reference (a component never extends beyond its bounding box).
    Review 11 Sep: 17.5 s -> 0.18 s per case. --selbsttest checks equality.
    Returns {(thr, mind): (hits row, clump count)}."""
    m0 = (p > min(THRESH_GRID)) | (lab > 0)
    zeilen = {}
    if not m0.any():
        leer = hits(lab, np.zeros(lab.shape, bool))
        return {(t, mind): (dict(leer), 0) for t in THRESH_GRID for mind in SIZE_GRID}
    sl = ndimage.find_objects(m0.astype(np.uint8))[0]
    pc, lc = p[sl], lab[sl] > 0
    lr, nr = ndimage.label(lc, structure=N26)
    for t in THRESH_GRID:
        lp, npred = ndimage.label(pc > t, structure=N26)
        sizes = np.bincount(lp.ravel(), minlength=npred + 1)
        both = (lp > 0) & (lr > 0)
        paare = (np.unique(np.stack([lp[both], lr[both]], 1), axis=0) if both.any()
                 else np.zeros((0, 2), np.int64))
        klumpen = int((sizes[1:] > MAX_KOMP_VOX).sum())
        for mind in SIZE_GRID:
            keep = (sizes >= mind) & (sizes <= MAX_KOMP_VOX); keep[0] = False
            n_keep = int(keep.sum())
            gp = paare[keep[paare[:, 0]]] if len(paare) else paare
            tp = int(len(np.unique(gp[:, 1]))) if len(gp) else 0
            fp = n_keep - (int(len(np.unique(gp[:, 0]))) if len(gp) else 0)
            zeilen[(t, mind)] = (dict(n_ref=int(nr), n_pred=n_keep, tp=tp, fp=fp, fn=int(nr) - tp,
                                      case_ref=int(nr > 0), case_pred=int(n_keep > 0),
                                      count_error=abs(n_keep - int(nr))), klumpen)
    return zeilen

def waehle(alle_zeilen):
    """alle_zeilen: list per case of gitter_zeilen(). -> (thr, mind, summary, klumpen)."""
    best = None
    for t in THRESH_GRID:
        for mind in SIZE_GRID:
            s = summarize([z[(t, mind)][0] for z in alle_zeilen])
            if best is None or s["f1"] > best[2]["f1"] + 1e-12:
                best = (t, mind, s, sum(z[(t, mind)][1] for z in alle_zeilen))
    if best[2]["f1"] <= 0:
        t, mind = FALLBACK
        best = (t, mind, summarize([z[(t, mind)][0] for z in alle_zeilen]),
                sum(z[(t, mind)][1] for z in alle_zeilen))
    return best

# ---------------------------------------------------------------- Fold
def neues_modell(P):
    nch = 1 if P["kanaele"] == "t2s" else 2
    m = AttentionUNet3D(in_channels=nch, basis=P["basis"], tiefe=P["tiefe"], gates=P["gates"],
                        dropout=P["dropout"], norm=P["norm"]).to(DEV)
    return m, nch

def zeit_pruefen(t0, limit, was):
    if limit and time.time() - t0 > limit:
        raise RuntimeError(f"Zeitzaun {limit:.0f}s erreicht vor {was}")

def train_falte(konf, k, outd, *, max_ep=MAX_EPOCHEN, min_ep=MIN_EPOCHEN, eval_jede=EVAL_JEDE,
                geduld=GEDULD, n_patches=None, zeit_limit=FOLD_TIME_MAX_S):
    P = konf_params(konf)
    if n_patches:
        P["n_patches"] = n_patches
    folds = json.load(open(f"{ROOT}/dev/split.json"))["folds"]
    recs = FA["recs"]
    i_f = (k + 1) % 5
    test_ids = [s for s in folds[k] if s in recs]
    val_ids = [s for s in folds[i_f] if s in recs]
    train_ids = [s for j in range(5) if j not in (k, i_f) for s in folds[j] if s in recs]
    assert test_ids and val_ids and train_ids and not (set(test_ids) & set(train_ids + val_ids))
    t0 = time.time()
    torch.manual_seed(SEED + k); rng = np.random.RandomState(SEED + k)   # reproducible per fold
    model, nch = neues_modell(P)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    loss_fn = VERLUSTE[P["loss"]]
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    verlauf, best, ohne = [], None, 0
    for ep in range(1, max_ep + 1):
        zeit_pruefen(t0, zeit_limit, f"Epoche {ep}")
        model.train(); t_ep = time.time(); verl = []
        xb, yb = stichprobe(train_ids, rng, P)
        for b0 in range(0, len(xb), P["bs"]):
            x2, y2 = xb[b0:b0 + P["bs"]].to(DEV), yb[b0:b0 + P["bs"]].to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = loss_fn(model(x2), y2)
            loss.backward(); opt.step(); verl.append(float(loss))
        sched.step()
        zeile = dict(epoche=ep, loss=round(float(np.mean(verl)), 4), train_s=round(time.time() - t_ep, 1))
        if ep % eval_jede == 0 or ep == max_ep:
            zeit_pruefen(t0, zeit_limit, f"innere Auswertung ep {ep}")
            t_k = time.time()
            z = [gitter_zeilen(proba_fall(model, s, nch=nch)[0], recs[s]["lab"]) for s in val_ids]
            thr, mind, s, kl = waehle(z)
            zeile.update(innen_f1=s["f1"], innen_sens=s["sensitivity"], innen_prec=s["precision"],
                         innen_fp_fall=s["fp_per_case"], schwelle=thr, mindestgroesse=mind,
                         klumpen=kl, karten_s=round(time.time() - t_k, 1))
            if LETZTER_STAND or best is None or s["f1"] > best["f1"] + 1e-9:     # LETZTER_STAND: every state replaces the previous one -> the last one remains at the end
                best = dict(f1=s["f1"], epoche=ep, schwelle=thr, mindestgroesse=mind, summary=s,
                            state={n: t.detach().cpu().clone() for n, t in model.state_dict().items()})
                ohne = 0
            else:
                ohne += 1
            print(f"[{konf} f{k}] ep {ep:2d} loss {zeile['loss']:.3f}  inner F1 {s['f1']:.3f} "
                  f"(P {s['precision']:.2f} S {s['sensitivity']:.2f} thr {thr} mind {mind} "
                  f"clumps {kl})  [{time.time() - t0:.0f}s]", flush=True)
            verlauf.append(zeile)
            if ep >= min_ep and ohne >= geduld:
                print(f"[{konf} f{k}] early stop after ep {ep} (best ep {best['epoche']})", flush=True)
                break
        else:
            verlauf.append(zeile)
    json.dump(verlauf, open(f"{outd}/verlauf.json", "w"), indent=1)
    model.load_state_dict(best["state"])
    thr, mind = best["schwelle"], best["mindestgroesse"]

    zeit_pruefen(t0, zeit_limit, "Testfalte")
    rows, klumpen_test, n_kacheln = [], {}, 0
    zur = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))            # (z,y,x) -> (x,y,z)
    for sid in test_ids:
        r = recs[sid]
        p, nk = proba_fall(model, sid, nch=nch); n_kacheln += nk
        pb = filtert_m(p, thr, mind, klumpen_test)
        row = dict(id=sid, fold=k, schwelle=thr, mindestgroesse=mind, **hits(r["lab"], pb))
        rows.append(row)
        nib.save(nib.Nifti1Image(zur(pb.astype(np.uint8)), r["affine"]), f"{outd}/{sid}_pred.nii.gz")
        nib.save(nib.Nifti1Image(zur(np.round(p * 255).astype(np.uint8)), r["affine"]),
                 f"{outd}/{sid}_pred_proba.nii.gz")
        json.dump(row, open(f"{outd}/{sid}_pred_meta.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{outd}/modell.pt")
    s_test = summarize(rows)
    meta = dict(konf=konf, trial=konf, falte=k, innere_falte=i_f, ok=True, params=P, nch=nch,
                epochs=verlauf[-1]["epoche"], epoche_gewaehlt=best["epoche"],
                schwelle=float(thr), mindestgroesse=int(mind), f1_innen=best["f1"],
                innen_summary=best["summary"], f1_testfalte=s_test["f1"], test_summary=s_test,
                max_komp_vox=MAX_KOMP_VOX, rand_versatz=RAND_VERSATZ, klumpen_testfalte=klumpen_test,
                gitter=os.path.basename(GITTER), n_train=len(train_ids), n_innen=len(val_ids),
                n_test=len(test_ids), kacheln_test=n_kacheln, dauer_s=round(time.time() - t0, 1),
                vram_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if DEV == "cuda" else None)
    json.dump(meta, open(f"{outd}/meta.json", "w"), indent=1)
    print(f"[{konf} f{k}] DONE: test F1 {s_test['f1']} (P {s_test['precision']} S "
          f"{s_test['sensitivity']}, thr {thr}, mind {mind}, ep {best['epoche']}, "
          f"clumps {klumpen_test.get('klumpen', 0)}) {meta['dauer_s']:.0f}s", flush=True)
    return meta

def falte_ordner(konf, k):
    """Only a fold with meta.ok counts as done. Everything else is set aside, so
    maps from an old model and metadata from a new one are never mixed."""
    outd = f"{ROOT}/ergebnisse/cv5-{konf}/fold{k}"
    mp = f"{outd}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("ok"):
        return outd, True
    if os.path.isdir(outd) and os.listdir(outd):
        ab = f"{outd}_abgebrochen-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(outd, ab)
        print(f"[{konf} f{k}] incomplete fold set aside: {ab}", flush=True)
    os.makedirs(outd, exist_ok=True)
    return outd, False

# ---------------------------------------------------------------- Self-test (CPU)
def selbsttest():
    """gitter_zeilen == metric.hits(filtert_m(...)) on random volumes (also with clumps);
    zufalls_start keeps the lesion within the patch."""
    rng = np.random.RandomState(0)
    for it in range(12):
        shp = (40, 70, 90)
        lab = np.zeros(shp, np.uint8)
        for _ in range(rng.randint(0, 6)):
            c = [rng.randint(3, s - 3) for s in shp]; r = rng.randint(1, 3)
            lab[c[0] - r:c[0] + r, c[1] - r:c[1] + r, c[2] - r:c[2] + r] = 1
        p = ndimage.gaussian_filter(rng.rand(*shp).astype(np.float32), 1.5)
        p = (p - p.min()) / (p.max() - p.min())
        if it % 3 == 0:
            p[5:35, 10:60, 10:80] = 0.95                    # clump over several references
        p = np.clip(p + 0.5 * ndimage.gaussian_filter(lab.astype(np.float32), 1), 0, 1)
        z = gitter_zeilen(p, lab)
        for t in THRESH_GRID:
            for mind in SIZE_GRID:
                a = hits(lab, filtert_m(p, t, mind)); b = z[(t, mind)][0]
                assert a == b, (it, t, mind, a, b)
    for _ in range(20000):
        shp = (60, 80, 120); c = [rng.randint(s) for s in shp]; st = zufalls_start(c, shp, rng)
        assert all(st[i] <= c[i] < st[i] + PATCH[i] for i in range(3)), (c, st)
    print("selbsttest: gitter_zeilen == hits(filtert_m) on 12 volumes x 28 cells; "
          "zufalls_start 20000/20000 within the patch -- ok")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="one fold complete, mini settings")
    ap.add_argument("--selbsttest", action="store_true")
    ap.add_argument("--konf", nargs="*", default=list(KONFS))
    ap.add_argument("--falten", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--zeit", type=int, default=None, help="total time limit in s")
    a = ap.parse_args()
    if a.selbsttest:
        selbsttest(); return
    np.random.seed(SEED)
    t_start = time.time()
    if a.probe:
        # Every path of the full fold is run through once (on 11 Sep every fold died on a
        # call that the old probe never executed). Mini: fold 0, 4 epochs, 2 evaluations.
        lade_daten()
        outd = f"{ROOT}/ergebnisse/cv5-probe/fold0"
        if os.path.isdir(outd):
            shutil.rmtree(outd)
        os.makedirs(outd)
        t0 = time.time()
        meta = train_falte("r1", 0, outd, max_ep=4, min_ep=4, eval_jede=2, n_patches=160)
        verl = json.load(open(f"{outd}/verlauf.json"))
        ep_s = float(np.mean([v["train_s"] for v in verl])) * REZEPT["n_patches"] / 160
        karte_s = float(np.mean([v["karten_s"] for v in verl if "karten_s" in v])) / max(meta["n_innen"], 1)
        n_eval = MAX_EPOCHEN // EVAL_JEDE
        falte_s = MAX_EPOCHEN * ep_s + n_eval * meta["n_innen"] * karte_s + meta["n_test"] * karte_s
        out = dict(epoche_s_voll=round(ep_s, 1), karte_fall_s=round(karte_s, 2),
                   schatz_falte_s=round(falte_s), schatz_konf_h=round(5 * falte_s / 3600, 2),
                   schatz_alle_h=round(len(KONFS) * 5 * falte_s / 3600, 2),
                   probe_dauer_s=round(time.time() - t0, 1), vram_gib=meta["vram_gib"],
                   probe_meta={k: meta[k] for k in ("f1_testfalte", "schwelle", "mindestgroesse",
                                                     "epoche_gewaehlt", "klumpen_testfalte")})
        json.dump(out, open(f"{ROOT}/dev/cv5_probe.json", "w"), indent=1)
        print("PROBE DONE:", json.dumps(out, ensure_ascii=False), flush=True)
        return

    lade_daten()
    status_p = f"{ROOT}/dev/cv5_status.json"
    status = json.load(open(status_p)) if os.path.exists(status_p) else {}
    t_end = t_start + a.zeit if a.zeit else None
    fehler = 0
    for konf in a.konf:
        for k in a.falten:
            if t_end and time.time() > t_end:
                print(f"Time limit {a.zeit}s -- rest in the next run (resumable).", flush=True)
                json.dump(status, open(status_p, "w"), indent=1)
                sys.exit(3)
            outd, fertig = falte_ordner(konf, k)
            if fertig:
                print(f"[{konf} f{k}] done -- skipped", flush=True); continue
            t0 = time.time()
            try:
                lim = min(FOLD_TIME_MAX_S, t_end - t0) if t_end else FOLD_TIME_MAX_S
                m = train_falte(konf, k, outd, zeit_limit=lim)
                status[f"{konf}/fold{k}"] = dict(ok=True, f1=m["f1_testfalte"], dauer_s=m["dauer_s"])
            except Exception as e:
                fehler += 1
                if DEV == "cuda":
                    torch.cuda.empty_cache()
                status[f"{konf}/fold{k}"] = dict(ok=False, fehler=f"{type(e).__name__}: {e}"[:300],
                                                 dauer_s=round(time.time() - t0, 1))
                print(f"[{konf} f{k}] ERROR: {type(e).__name__}: {e}", flush=True)
            json.dump(status, open(status_p, "w"), indent=1)
    print(f"CV5 DONE: {fehler} fold(s) with an error. Status: dev/cv5_status.json", flush=True)
    sys.exit(1 if fehler else 0)

if __name__ == "__main__":
    main()
