#!/usr/bin/env python3
"""Architecture tournament on VALDO T2* (Claude 18 Sep 2026, user task).

Plugs the networks from code/netze.py into the UNCHANGED cv5 pipeline: the same five folds
(dev/split.json), the same grid C, the same sampling, the same inner fold for epoch and
threshold, the same clump guard, the same metric (metric.hits, 26-connectivity components). Per
network ONLY the architecture changes -- recipe = cv5 r3 (T2* + FRST, InstanceNorm, BCE+Dice,
AdamW 3e-4 cosine, 800 patches/epoch, batch 4). The validation set is never read.

  python turnier.py --liste
  python turnier.py --probe [--netz a04-2p5d ...] [--bs 1] [--vram-anteil 0.06]
        Mini fold per network on the GPU (2 epochs, 2 evaluations, 40 patches): every path
        once, including bfloat16 autocast. Writes to ergebnisse/turnier-probe/, never into the tournament.
  python turnier.py --netz a01-attunet [--falten 0 1 2 3 4] [--seed 42] [--kanaele t2s]
        Results: ergebnisse/turnier/<netz>[-t2s][-s<seed>]/fold<k>/ (layout like cv5),
        Status: dev/turnier_status.json. Resumable: finished folds are skipped.
Return value: 0 all ok, 1 at least one fold with an error.

Deviations from the recipe that can only occur here are recorded in meta.json and in the status:
  bs_abweichend  CUDA OOM at batch 4 -> retry with batch 2 (large networks only)
  vortrainiert   number of loaded encoder tensors (a08-swin only)
"""
import os, sys, json, time, shutil, argparse, functools
import numpy as np
import torch

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
import cv5
from netze import NETZE, parameterzahl

FOLD_ZEIT_S = 6 * 3600          # time fence per fold; cv5 has 3 h, Swin with checkpointing is slower
_LETZTES = {}                   # the last built model (for parameter count/vortrainiert in meta)


def loss_bce_dice_ds(lg, ziel):
    """Like cv5.loss_bce_dice. If the network delivers deep-supervision heads (B, K, 1, D, H, W),
    they are weighted 1, 1/2, 1/4, ... (nnU-Net). In eval() only the main head is ever used."""
    if lg.dim() == 5:
        return cv5.loss_bce_dice(lg, ziel)
    gew = [0.5 ** i for i in range(lg.shape[1])]
    return sum(g * cv5.loss_bce_dice(lg[:, i], ziel) for i, g in enumerate(gew)) / sum(gew)


BLOB_GEWICHT = 0.5             # global + 0.5 * blob  ==  Kofler's alpha=2, beta=1 up to the factor 2


def loss_blob(lg, ziel):
    """blob loss (Kofler et al., arXiv:2205.08209; confirmed at the source on 23 Sep 2026: IPMI 2023, Springer LNCS) on BCE+Dice: on top of the
    global loss, each bleed gets its own term, computed on the patch WITHOUT the voxels of the other bleeds. This way
    a 4-voxel bleed counts as much as one with 80 voxels; in the global dice it would be drowned out. Components with
    26-connectivity as in the metric. Patches without a bleed carry only the global term. With
    deep-supervision heads, the blob term acts only on the main head. Computed in float32 (small sums)."""
    import torch.nn.functional as F
    from scipy import ndimage
    basis = loss_bce_dice_ds(lg, ziel)
    haupt = (lg if lg.dim() == 5 else lg[:, 0]).float()
    z = ziel.float()
    terme = []
    for b, vol in enumerate(z[:, 0].detach().cpu().numpy() > 0.5):
        if not vol.any():
            continue
        marken, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
        marken = torch.from_numpy(marken).to(z.device)
        lb, zb = haupt[b, 0], z[b, 0]
        bce_voxel = F.binary_cross_entropy_with_logits(lb, zb, reduction="none")
        p = torch.sigmoid(lb)
        for j in range(1, n + 1):
            m = ((marken == j) | (marken == 0)).float()      # this bleed + background, the others masked out
            bce = (bce_voxel * m).sum() / m.sum()
            dice = 1 - (2 * (p * zb * m).sum() + 1.0) / ((p * m).sum() + (zb * m).sum() + 1.0)
            terme.append(0.5 * bce + 0.5 * dice)
    if not terme:
        return basis
    return basis + BLOB_GEWICHT * torch.stack(terme).mean()


ZENTRUM_SIGMA_MM = 2.0          # Gaussian width of the center map in mm (CenSynCMB: 2 voxels at 1 mm isotropic)
ZENTRUM_GAMMA, ZENTRUM_EPS = 2.5, 0.05   # weight (C + eps)^gamma; CenSynCMB states eps = 1e-3, see loss_zentrum
ZENTRUM_GEWICHT = 1.0
GITTER_MM = (1.0, 0.5, 0.5)     # voxel size of the training grid in (z, y, x)


def baue_aniso_mit_zentrum(nch):
    """a03-aniso with a second head for a CENTER MAP (He et al. 2026, CenSynCMB, arXiv 2607.05325: the biggest
    single contribution in their ablation, +4.5 F1 points). The head sits on the same decoder features as the
    segmentation head (forward hook on `aus`). During training the network delivers (B, 2, D, H, W) = segmentation +
    center; in eval() ONLY the segmentation -- evaluation, TTA, ensemble and second stage remain unchanged."""
    import torch.nn as nn
    from netze import AnisoUNet3D, Polster

    class _AnisoMitZentrum(AnisoUNet3D):
        def __init__(self, nch):
            super().__init__(nch)
            self.zentrum = nn.Conv3d(self.aus.in_channels, 1, 1)
            self._merkmale = None
            self.aus.register_forward_pre_hook(lambda modul, eingabe: setattr(self, "_merkmale", eingabe[0]))

        def forward(self, x):
            seg = super().forward(x)
            if not self.training:
                return seg
            return torch.cat([seg, self.zentrum(self._merkmale)], 1)

    return Polster(_AnisoMitZentrum(nch), (8, 16, 16))      # same padding as a03-aniso


def zentrumskarte(ziel):
    """(B, 1, D, H, W) label -> Gaussian bump per bleed at the centroid, C = max_k exp(-d_k^2 / (2 sigma^2)), d in mm."""
    from scipy import ndimage
    karte = torch.zeros_like(ziel, dtype=torch.float32)
    achsen = [torch.arange(n, device=ziel.device, dtype=torch.float32) * mm for n, mm in zip(ziel.shape[2:], GITTER_MM)]
    for b, vol in enumerate(ziel[:, 0].detach().cpu().numpy() > 0.5):
        if not vol.any():
            continue
        marken, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
        for cz, cy, cx in ndimage.center_of_mass(vol, marken, range(1, n + 1)):
            d2 = ((achsen[0] - cz * GITTER_MM[0]) ** 2)[:, None, None] + ((achsen[1] - cy * GITTER_MM[1]) ** 2)[None, :, None] \
                + ((achsen[2] - cx * GITTER_MM[2]) ** 2)[None, None, :]
            karte[b, 0] = torch.maximum(karte[b, 0], torch.exp(-d2 / (2 * ZENTRUM_SIGMA_MM ** 2)))
    return karte


def loss_zentrum(lg, ziel):
    """BCE+Dice on the segmentation head + focal MSE on the center map (CenSynCMB). Deliberate deviations from the paper:
    (1) normalized to the weight sum instead of the voxel count, and a fixed weight of 1 instead of learned uncertainty -- our
    loss does not see the network parameters; (2) eps = 0.05 instead of 1e-3: with 1e-3 the background practically doesn't count, with
    0.05 the head learns a usable detection map (usable later as a second opinion); (3) sigma in mm, because our
    grid is 0.5 x 0.5 x 1 mm. Patches without a bleed: target map 0, the head learns 'no center' there."""
    if lg.dim() != 5 or lg.shape[1] != 2:
        raise RuntimeError("--verlust zentrum needs the network a12-anisozentrum (two output channels during training)")
    basis = cv5.loss_bce_dice(lg[:, :1], ziel)
    karte = zentrumskarte(ziel)
    gewicht = (karte + ZENTRUM_EPS) ** ZENTRUM_GAMMA
    fehler = (torch.sigmoid(lg[:, 1:2].float()) - karte) ** 2
    return basis + ZENTRUM_GEWICHT * (gewicht * fehler).sum() / gewicht.sum()


FNRW_GAMMA, FNRW_RHO, FNRW_GRENZEN = 0.5, 1.5, (0.5, 5.0)     # values from CenSynCMB (He et al. 2026)
_FNRW = {"mittlere_groesse": None}


def loss_fnrw(lg, ziel):
    """BCE+Dice PER PATCH, weighted by the bleeds it contains (CenSynCMB: 'false-negative-driven reweighting'):
    w_k = (mean size / size_k)^0.5 * (1 + 1.5 * (1 - r_k)), r_k = highest current network response on the bleed.
    Small and only weakly detected bleeds count for more; patches without a bleed keep weight 1. The paper does not say
    how several bleeds in one patch combine -- here the MAXIMUM (the hardest one determines the
    weight), clamped to [0.5, 5]. With all weights equal to 1, the value is exactly cv5.loss_bce_dice."""
    import torch.nn.functional as F
    from scipy import ndimage
    if lg.dim() != 5 or lg.shape[1] != 1:
        raise RuntimeError("--verlust fnrw expects a network with a single output channel and no deep supervision")
    if _FNRW["mittlere_groesse"] is None:               # scale constant from the loaded cases (no label information about test cases needed)
        groessen = [len(k) for r in cv5.FA["recs"].values() for k in r["komp"]]
        _FNRW["mittlere_groesse"] = float(np.mean(groessen)) if groessen else 60.0
    p = torch.sigmoid(lg)
    bce = F.binary_cross_entropy_with_logits(lg, ziel, reduction="none").flatten(1).mean(1)
    pf, zf = p.flatten(1), ziel.flatten(1)
    dice = 1 - (2 * (pf * zf).sum(1) + 1.0) / (pf.sum(1) + zf.sum(1) + 1.0)
    je_ausschnitt = 0.5 * bce + 0.5 * dice
    antwort = p.detach().float().cpu().numpy()[:, 0]
    gewichte = []
    for b, vol in enumerate(ziel[:, 0].detach().cpu().numpy() > 0.5):
        w = 1.0
        if vol.any():
            marken, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
            groesse = np.bincount(marken.ravel(), minlength=n + 1)[1:]
            r = ndimage.maximum(antwort[b], marken, range(1, n + 1))
            w_k = (_FNRW["mittlere_groesse"] / groesse) ** FNRW_GAMMA * (1 + FNRW_RHO * np.clip(1 - np.asarray(r), 0, 1))
            w = float(np.clip(w_k.max(), *FNRW_GRENZEN))
        gewichte.append(w)
    gewichte = torch.tensor(gewichte, device=lg.device, dtype=je_ausschnitt.dtype)
    return (gewichte * je_ausschnitt).sum() / gewichte.sum()


HN_START, HN_JEDE, HN_ANTEIL = 10, 10, 0.6       # mine from epoch 11 every 10 epochs; replace 60 % of the bleed-free patches
_HN = {"aufrufe": 0, "zentren": []}


def fehlalarm_zentren(modell, train_ids, nch):
    """The CURRENT network predicts its own training cases (fixed post-processing 0.3 / 8 voxels); centroids of
    components that touch no reference become sampling centers. Strictly within the fold: no other run,
    no test case."""
    from scipy import ndimage
    zentren = []
    for sid in train_ids:
        r = cv5.FA["recs"][sid]
        maske = cv5.filtert_m(cv5.proba_fall(modell, sid, nch=nch)[0], 0.3, 8)
        marken, n = ndimage.label(maske, structure=cv5.N26)
        if n == 0:
            continue
        trifft = set(np.unique(marken[r["lab"] > 0]).tolist()) - {0}
        for k, c in enumerate(ndimage.center_of_mass(maske, marken, range(1, n + 1)), start=1):
            if k not in trifft:
                zentren.append((sid, tuple(int(round(v)) for v in c)))
    return zentren


def mit_harten_negativen(stichprobe):
    """Online hard negative mining in the sampling (related to Dou et al. 2016, but there the SECOND stage learns on the false positives of the first): part of the bleed-free patches
    are replaced with patches around the network's own false positives. The reference in the patch stays the real one --
    if a bleed lies next to the false positive, it is correctly marked in the label."""
    def neu(train_ids, rng, P):
        _HN["aufrufe"] += 1
        ep = _HN["aufrufe"]
        modell = _LETZTES.get("modell")
        nch = 1 if P["kanaele"] == "t2s" else 2
        if modell is not None and ep > HN_START and (ep - HN_START - 1) % HN_JEDE == 0:
            t0 = time.time()
            _HN["zentren"] = fehlalarm_zentren(modell, train_ids, nch)
            modell.train()                                    # proba_fall schaltet auf eval()
            print(f"[hard negatives] epoch {ep}: {len(_HN['zentren'])} false-positive centers in "
                  f"{len({s for s, _ in _HN['zentren']})} of {len(train_ids)} training cases ({time.time() - t0:.0f} s)", flush=True)
        xs, ys = stichprobe(train_ids, rng, P)
        if _HN["zentren"]:
            leer = [j for j in range(len(ys)) if float(ys[j].sum()) == 0.0]
            for j in leer[:int(len(leer) * HN_ANTEIL)]:       # order is already shuffled (permutation in cv5.stichprobe)
                sid, c = _HN["zentren"][rng.randint(len(_HN["zentren"]))]
                r = cv5.FA["recs"][sid]; st = cv5.zufalls_start(c, r["img"].shape, rng)
                bild = np.stack([cv5.schnitt(r["img"], st)] + ([cv5.schnitt(r["frs"], st)] if nch == 2 else []), 0)
                label = cv5.schnitt(r["lab"], st)[None].astype(np.float32)
                if P["augment"]:
                    for ax in (1, 2, 3):
                        if rng.rand() < 0.5:
                            bild, label = np.flip(bild, ax), np.flip(label, ax)
                xs[j], ys[j] = torch.from_numpy(np.ascontiguousarray(bild)), torch.from_numpy(np.ascontiguousarray(label))
        return xs, ys
    return neu


# ---------------------------------------------------------------- Synthetic bleeds and vessel mimics (28 Sep 2026, WORKLOG.md 5ae)
# cmb/synthesis/build_synthetic_grid.py writes a grid in which every case carries additional synthetic bleeds (in the label)
# and vessel mimics (image only), with FRST recomputed. Here ONLY the training cases of the fold are swapped for this
# version while the sample is drawn; the test and inner selection folds stay real, and after drawing,
# the real data is back in cv5.FA -- proba_fall (choice of checkpoint, test map) never sees a synthetic voxel.
_SYN = {"gitter": None, "recs": {}, "ids": None}


def _lade_syn(gitter, sid):
    import nibabel as nib
    from scipy import ndimage
    if True:
        if sid not in _SYN["recs"]:
            d = f"{gitter}/{sid}"
            dreh = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
            ni = nib.load(f"{d}/{sid}_image.nii.gz")
            img = dreh(np.asarray(ni.dataobj, dtype=np.float32))
            frs = dreh(np.asarray(nib.load(f"{d}/{sid}_frst.nii.gz").dataobj, dtype=np.float32))
            lab = dreh((np.asarray(nib.load(f"{d}/{sid}_label.nii.gz").dataobj) > 0).astype(np.uint8))
            echt = cv5.FA["recs"][sid]
            assert img.shape == echt["img"].shape, f"{sid}: synthetic grid does not match the training grid"
            assert (lab >= echt["lab"]).all(), f"{sid}: synthetic label does not contain the real one"
            ll, n = ndimage.label(lab, structure=cv5.N26)
            syn = dreh(np.asarray(nib.load(f"{d}/{sid}_synth.nii.gz").dataobj) == 1)         # 1 = synthetic bleed
            ls, ns = ndimage.label(syn, structure=cv5.N26)
            _SYN["recs"][sid] = dict(img=img, frs=frs, lab=lab, komp=[np.argwhere(ll == j) for j in range(1, n + 1)],
                                     komp_syn=[np.argwhere(ls == j) for j in range(1, ns + 1)],
                                     fold=echt["fold"], affine=echt["affine"])
        return _SYN["recs"][sid]


def mit_synthese(stichprobe, gitter):
    _SYN["gitter"] = gitter
    lade = lambda sid: _lade_syn(gitter, sid)

    def neu(train_ids, rng, P):
        ids = tuple(sorted(train_ids))
        if _SYN["ids"] != ids:                              # new fold: free the cache of the old fold
            _SYN["recs"] = {k: v for k, v in _SYN["recs"].items() if k in ids}
            _SYN["ids"] = ids
            print(f"[synthesis] {len(ids)} training cases from {gitter}", flush=True)
        recs = cv5.FA["recs"]
        echt = {s: recs[s] for s in train_ids}
        try:
            for s in train_ids:
                recs[s] = lade(s)
            return stichprobe(train_ids, rng, P)
        finally:
            for s in train_ids:
                recs[s] = echt[s]
    return neu


def mit_synthese_anteil(stichprobe, gitter, anteil):
    """Second version (WORKLOG.md 5ae-1b, 28 Sep 2026): the budget of REAL lesions stays exactly as in the base -- the
    sample is drawn unchanged; afterwards the fraction `anteil` of the patches WITHOUT a lesion is replaced by patches around
    synthetic bleeds from the synthesis grid (image, FRST and label of the synthetic version, i.e. with the
    synthetic bleeds in the label). This version has no vessel mimics. Test and selection folds stay real."""
    _SYN["gitter"] = gitter

    def neu(train_ids, rng, P):
        xs, ys = stichprobe(train_ids, rng, P)
        ids = tuple(sorted(train_ids))
        if _SYN["ids"] != ids:
            _SYN["recs"] = {k: v for k, v in _SYN["recs"].items() if k in ids}
            _SYN["ids"] = ids
            print(f"[synthesis fraction {anteil}] {len(ids)} training cases from {gitter}", flush=True)
        nch = 1 if P["kanaele"] == "t2s" else 2
        pool = []
        for s in train_ids:
            r = _lade_syn(gitter, s)
            pool += [(s, j) for j in range(len(r["komp_syn"]))]
        if not pool:
            return xs, ys
        leer = [j for j in range(len(ys)) if float(ys[j].sum()) == 0.0]
        for j in leer[:int(len(leer) * anteil)]:
            sid, k = pool[rng.randint(len(pool))]
            r = _lade_syn(gitter, sid); vox = r["komp_syn"][k]; c = vox[rng.randint(len(vox))]
            st = cv5.zufalls_start(c, r["img"].shape, rng)
            bild = np.stack([cv5.schnitt(r["img"], st)] + ([cv5.schnitt(r["frs"], st)] if nch == 2 else []), 0)
            label = cv5.schnitt(r["lab"], st)[None].astype(np.float32)
            if P["augment"]:
                for ax in (1, 2, 3):
                    if rng.rand() < 0.5:
                        bild, label = np.flip(bild, ax), np.flip(label, ax)
            xs[j], ys[j] = torch.from_numpy(np.ascontiguousarray(bild)), torch.from_numpy(np.ascontiguousarray(label))
        return xs, ys
    return neu


# ---------------------------------------------------------------- Tissue map as input channels (21 Sep 2026)
# SynthSeg label -> class 1..5. So far we only use the map AFTER the prediction (CSF rule, +0.06 on VALDO). As an
# input, the network can learn for itself where mimics sit (sulci, ventricle rim, basal ganglia) -- information
# different from the T2* image, without an extra sequence on the target cohort (there SynthSeg --robust runs directly on T2*).
GEWEBE_KLASSEN = {}
for _l in (4, 5, 14, 15, 24, 43, 44):                          GEWEBE_KLASSEN[_l] = 1      # CSF
for _l in (3, 42, 17, 18, 53, 54):                             GEWEBE_KLASSEN[_l] = 2      # cortex + mesiotemporal
for _l in (2, 41):                                             GEWEBE_KLASSEN[_l] = 3      # cerebral white matter
for _l in (10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60):    GEWEBE_KLASSEN[_l] = 4      # deep nuclei
for _l in (7, 8, 16, 46, 47):                                  GEWEBE_KLASSEN[_l] = 5      # cerebellum + brainstem
N_GEWEBE = 5
_GEWEBE = {"an": False}


def mit_gewebekanal(lade, muster):
    """Attaches `gew` (uint8, 0..5, (z, y, x)) to every case. Maps on a different grid (second cohort: 1 mm) are put
    onto the training grid with nearest neighbor (the same function as in cmb.analysis)."""
    import nibabel as nib
    sys.path.insert(0, ROOT)
    from cmb.analysis.fp_location import load_synthseg, synthseg_path
    tabelle = np.zeros(256, np.uint8)
    for label, klasse in GEWEBE_KLASSEN.items():
        tabelle[label] = klasse

    def neu(*a, **kw):
        lade(*a, **kw)
        anteile = np.zeros(N_GEWEBE + 1)
        for sid, r in cv5.FA["recs"].items():
            pfad = synthseg_path(muster, sid)
            assert os.path.exists(pfad), f"tissue map missing: {pfad}"
            wie = nib.Nifti1Image(np.zeros(r["img"].shape[::-1], np.uint8), r["affine"])      # grid of the case in (x, y, z)
            seg = load_synthseg(pfad, wie)
            gew = tabelle[np.clip(seg, 0, 255)].transpose(2, 1, 0)
            assert gew.shape == r["img"].shape, sid
            r["gew"] = np.ascontiguousarray(gew)
            anteile += np.bincount(r["gew"][r["img"] != 0].ravel(), minlength=N_GEWEBE + 1) / max(int((r["img"] != 0).sum()), 1)
        anteile /= max(len(cv5.FA["recs"]), 1)
        print("Tissue channels: fraction of the brain (none / CSF / cortex / white matter / deep / infratentorial): "
              + " / ".join(f"{v:.2f}" for v in anteile), flush=True)
    return neu


_ZUSATZ = {"namen": []}          # Arm 8 (29 Sep 2026): continuous extra channels <id>_<name>.nii.gz from the grid folder, e.g. t1 t2


def mit_zusatzkanaelen(lade, namen):
    """Attaches r["zus"] = [map per name] ((z, y, x), float32, same shape as the image) to every case. The maps sit
    already normalized in the case's grid folder (code/zusatzkanaele_bauen.py)."""
    import nibabel as nib

    def neu(*a, **kw):
        lade(*a, **kw)
        dreh = lambda arr: np.ascontiguousarray(arr.transpose(2, 1, 0))
        for sid, r in cv5.FA["recs"].items():
            r["zus"] = []
            for name in namen:
                pfad = f"{cv5.GITTER}/{sid}/{sid}_{name}.nii.gz"
                assert os.path.exists(pfad), f"extra channel missing: {pfad}"
                k = dreh(np.asarray(nib.load(pfad).dataobj, dtype=np.float32))
                assert k.shape == r["img"].shape, (sid, name, k.shape, r["img"].shape)
                r["zus"].append(k)
        print(f"Extra channels {namen}: {len(cv5.FA['recs'])} cases", flush=True)
    return neu


def _n_extra():
    return (N_GEWEBE if _GEWEBE["an"] else 0) + len(_ZUSATZ["namen"])


def _kanaele(r, st, nch):
    """Image (+ FRST) (+ five one-hot tissue channels) (+ continuous extra channels) for the patch starting at st."""
    teile = [cv5.schnitt(r["img"], st)]
    if nch >= 2:
        teile.append(cv5.schnitt(r["frs"], st))
    if _GEWEBE["an"]:
        g = cv5.schnitt(r["gew"], st)
        teile += [(g == k).astype(np.float32) for k in range(1, N_GEWEBE + 1)]
    for k in r.get("zus", []):
        teile.append(cv5.schnitt(k, st))
    return np.stack(teile, 0)


def stichprobe_gewebe(train_ids, rng, P):
    """WORD-FOR-WORD cv5.stichprobe (same order of random draws -> with the same seed, the same patches as
    the base), just with the extra channels."""
    recs = cv5.FA["recs"]
    komps = [(s, j) for s in train_ids for j in range(len(recs[s]["komp"]))]
    n = P["n_patches"]
    n_l, n_h = int(n * P["anteil_laesion"]), int(n * P["anteil_hirn"])
    nch = (1 if P["kanaele"] == "t2s" else 2) + _n_extra()
    xs, ys = np.empty((n, nch) + cv5.PATCH, np.float32), np.empty((n, 1) + cv5.PATCH, np.float32)
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
        r = recs[s]; st = cv5.zufalls_start(c, r["img"].shape, rng)
        p = _kanaele(r, st, nch)
        l = cv5.schnitt(r["lab"], st)[None].astype(np.float32)
        if P["augment"]:
            for ax in (1, 2, 3):
                if rng.rand() < 0.5:
                    p, l = np.flip(p, ax), np.flip(l, ax)
        xs[j], ys[j] = p, l
    idx = rng.permutation(n)
    return torch.from_numpy(xs[idx]), torch.from_numpy(ys[idx])


def proba_fall_gewebe(model, sid, *, bs=8, nch=2):
    """Like cv5.proba_fall (half stride, overlap averaging, empty tiles skipped), with the extra channels."""
    r = cv5.FA["recs"][sid]
    img = r["img"]; shp = img.shape; PATCH = cv5.PATCH
    padded = tuple(max(shp[i], PATCH[i]) for i in range(3))
    starts = []
    for i in range(3):
        st = list(range(0, padded[i] - PATCH[i] + 1, PATCH[i] // 2))
        if st[-1] + PATCH[i] < padded[i]:
            st.append(padded[i] - PATCH[i])
        starts.append(st)
    pred = np.zeros(padded, np.float32); cnt = np.zeros(padded, np.float32)
    kacheln = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    kacheln = [s for s in kacheln if cv5.schnitt(img, s).any()]
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cv5.DEV == "cuda"):
        for i in range(0, len(kacheln), bs):
            b = kacheln[i:i + bs]
            xb = np.stack([_kanaele(r, s, nch) for s in b])
            out = torch.sigmoid(model(torch.from_numpy(xb).to(cv5.DEV))).float().cpu().numpy()
            for k, (z, y, x) in enumerate(b):
                pred[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += out[k, 0]
                cnt[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += 1.0
    pred = (pred / np.maximum(cnt, 1))[:shp[0], :shp[1], :shp[2]]
    return pred, len(kacheln)


_INIT = {"muster": None, "fold": None}


def neues_modell(P):
    nch = (1 if P["kanaele"] == "t2s" else 2) + _n_extra()
    m = NETZE[P["netz"]](nch)
    if _INIT["muster"]:                                   # Feintuning: Gewichte derselben Falte als Start
        pfad = _INIT["muster"].format(fold=_INIT["fold"])
        m.load_state_dict(torch.load(pfad, map_location="cpu"))
        print(f"Starting weights loaded: {pfad}", flush=True)
    m = m.to(cv5.DEV)
    _LETZTES.update(parameter=parameterzahl(m), vortrainiert=getattr(m, "vortrainiert", None), modell=m)
    _HN.update(aufrufe=0, zentren=[])                     # harte Negative: je Falte neu schuerfen
    return m, nch


cv5.VERLUSTE["bce_dice_ds"] = loss_bce_dice_ds
cv5.VERLUSTE["blob"] = loss_blob
cv5.VERLUSTE["zentrum"] = loss_zentrum
cv5.VERLUSTE["fnrw"] = loss_fnrw
NETZE["a12-anisozentrum"] = baue_aniso_mit_zentrum
cv5.neues_modell = neues_modell
for _n in NETZE:
    cv5.KONFS[_n] = {"norm": "instance", "netz": _n, "loss": "bce_dice_ds"}


def mit_manifest(pfade):
    """Replaces cv5.lade_daten: cases from manifest lists (several cohorts possible). train_falte reads the folds from
    f"{cv5.ROOT}/dev/split.json" -- for this, cv5.ROOT is redirected to a helper folder with a matching split.json
    (cv5.ROOT is used in train_falte ONLY for that; turnier.py itself determines the result folders)."""
    import hashlib
    import nibabel as nib
    from scipy import ndimage
    eintraege = [e for p in pfade for e in json.load(open(p))]
    assert len({e["id"] for e in eintraege}) == len(eintraege), "case IDs must be unique across all manifests"
    hilfs = f"{ROOT}/dev/manifest_splits/" + hashlib.md5("|".join(sorted(pfade)).encode()).hexdigest()[:10]
    os.makedirs(f"{hilfs}/dev", exist_ok=True)
    json.dump({"folds": [[e["id"] for e in eintraege if e["fold"] == k] for k in range(5)], "manifeste": pfade},
              open(f"{hilfs}/dev/split.json", "w"))
    cv5.ROOT = hilfs

    def lade(ids=None):
        recs = {}
        for e in eintraege:
            if e["fold"] not in range(5) or (ids is not None and e["id"] not in ids):
                continue
            sid, d = e["id"], e["dir"]
            ni = nib.load(f"{d}/{sid}_image.nii.gz")
            dreh = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
            img = dreh(np.asarray(ni.dataobj, dtype=np.float32))
            frs = dreh(np.asarray(nib.load(f"{d}/{sid}_frst.nii.gz").dataobj, dtype=np.float32))
            lab = dreh((np.asarray(nib.load(f"{d}/{sid}_label.nii.gz").dataobj) > 0).astype(np.uint8))
            assert img.shape == frs.shape == lab.shape, sid
            ll, n = ndimage.label(lab, structure=cv5.N26)
            recs[sid] = dict(img=img, frs=frs, lab=lab, komp=[np.argwhere(ll == j) for j in range(1, n + 1)],
                             fold=e["fold"], affine=ni.affine)
        cv5.FA = dict(recs=recs)
        je = {}
        for e in eintraege:
            if e["id"] in recs:
                je[e.get("cohort", "?")] = je.get(e.get("cohort", "?"), 0) + 1
        print(f"Data from manifest: {len(recs)} cases {je}, {sum(len(r['komp']) for r in recs.values())} CMB components", flush=True)
    return lade


def mit_synthseg_maske(lade):
    """Sets image and FRST to 0 outside the SynthSeg brain (label > 0, dilated by 2 voxels). The labels
    stay untouched; if a reference CMB lies outside, this is counted and reported."""
    import nibabel as nib
    from scipy import ndimage
    def neu(*a, **kw):
        lade(*a, **kw)
        weg, cmb_aussen = [], 0
        for sid, r in cv5.FA["recs"].items():
            seg = np.asarray(nib.load(f"{ROOT}/dev/synthseg/{sid}_synthseg.nii.gz").dataobj) > 0
            m = ndimage.binary_dilation(np.ascontiguousarray(seg.transpose(2, 1, 0)), iterations=2)
            assert m.shape == r["img"].shape, sid
            vor = (r["img"] != 0).sum()
            r["img"][~m] = 0; r["frs"][~m] = 0
            weg.append(1 - (r["img"] != 0).sum() / max(vor, 1))
            cmb_aussen += sum(1 for k in r["komp"] if not m[tuple(k.T)].any())
        print(f"SynthSeg mask: on average {100 * np.mean(weg):.0f} % of the old mask removed; "
              f"reference CMBs entirely outside: {cmb_aussen}", flush=True)
    return neu


def mit_perzentil_normierung(lade, q=99.5):
    """Robust version of the original scale. As delivered, image = 1 - T2*/maximum; a single bright outlier
    then compresses the whole brain. Here: recompute T2*/maximum = 1 - image, divide by the q-th percentile of the
    brain voxels instead of the maximum, invert again, clip to [1e-6, 1]. Background stays 0."""
    def neu(*a, **kw):
        lade(*a, **kw)
        for r in cv5.FA["recs"].values():
            h = r["img"] != 0
            if h.any():
                roh = 1.0 - r["img"][h]
                p = max(float(np.percentile(roh, q)), 1e-6)
                r["img"][h] = np.clip(1.0 - roh / p, 1e-6, 1.0)
        print(f"Percentile normalization ({q}) applied per case in the brain", flush=True)
    return neu


LANDMARKEN_PERZENTILE = (10, 25, 50, 75, 90, 99)


def landmarken_bestimmen(bilder) -> list:
    """Reference landmarks: median of the per-case percentiles over the given images (on the DELIVERED, inverted scale)."""
    return list(np.median([np.percentile(b[b != 0], LANDMARKEN_PERZENTILE) for b in bilder], axis=0))


def mit_landmarken_normierung(lade, referenz_datei):
    """Piecewise linear alignment to fixed reference landmarks (Nyul & Udupa 1999), per case in the brain.

    WHY (measured 23 Sep 2026, cmb/analysis/harmonisation_check.py): the difference between our cohorts is
    NONLINEAR. All affine scales fail at this -- dividing by the maximum leaves the spread of the
    lesion contrast between cohorts at 0.060, the 99.5th percentile at 0.057, and z-normalization even makes it
    six times WORSE at 0.362 (dividing by the brain's spread aligns the distribution but distorts
    the lesion contrast differently per cohort in the process). The landmark alignment brings it down to 0.018 and aligns
    level and spread exactly -- the only method that can fix a nonlinear difference.

    How: per case, determine the percentiles 10/25/50/75/90/99 of the brain voxels and map them piecewise linearly onto the stored
    reference values (np.interp; outside the landmarks it continues constantly). Background stays exactly 0,
    brain voxels that would otherwise become 0 get 1e-6 -- the same convention as the other scales, so that
    "tile entirely outside" and "brain voxel" keep holding.

    The reference file MUST be the same as during training; otherwise the prediction is computed on a different scale.
    Generate: python code/landmarken_bauen.py --gitter dev/gitter_d --aus dev/landmarken_valdo.json
    """
    referenz = np.asarray(json.load(open(referenz_datei))["landmarken"], np.float64)
    assert len(referenz) == len(LANDMARKEN_PERZENTILE), referenz_datei

    def neu(*a, **kw):
        lade(*a, **kw)
        for r in cv5.FA["recs"].values():
            h = r["img"] != 0
            if h.any():
                quellen = np.percentile(r["img"][h], LANDMARKEN_PERZENTILE)
                if np.all(np.diff(quellen) > 0):          # np.interp braucht streng steigende Stuetzstellen
                    v = np.interp(r["img"][h], quellen, referenz).astype(np.float32)
                    v[v == 0] = 1e-6
                    r["img"][h] = v
        print(f"Landmark normalization applied per case in the brain (reference {os.path.basename(referenz_datei)})", flush=True)
    return neu


def mit_z_normierung(lade):
    """z-normalization of the image channel per case over the brain voxels (image != 0); background stays exactly 0,
    so that 'tile entirely outside' (proba_fall) and 'brain voxel' (stichprobe) keep holding. Brain voxels
    that would become exactly 0 through the normalization get 1e-6."""
    def neu(*a, **kw):
        lade(*a, **kw)
        for r in cv5.FA["recs"].values():
            h = r["img"] != 0
            if h.any():
                v = r["img"][h]; v = (v - v.mean()) / max(float(v.std()), 1e-6)
                v[v == 0] = 1e-6
                r["img"][h] = v
        print("z-normalization applied per case in the brain", flush=True)
    return neu


def mit_augmentierung(stichprobe):
    """After the recipe sampling (with flips), on the GPU, batch-wise: rotation about the z-axis
    +-15 degrees and scale 0.9-1.1 (image bilinear, label nearest neighbor -- a CMB of a few voxels
    must not be smeared into fractions), then only on image channel 0 and only in the brain:
    contrast x(0.9-1.1) around the patch mean, brightness +-0.1 std, noise up to 0.05 std."""
    import math
    import torch.nn.functional as F
    def neu(train_ids, rng, P):
        xs, ys = stichprobe(train_ids, rng, P)
        g = torch.Generator(device="cpu").manual_seed(int(rng.randint(2 ** 31 - 1)))
        aus_x, aus_y = [], []
        for b0 in range(0, len(xs), 32):
            x = xs[b0:b0 + 32].to(cv5.DEV); y = ys[b0:b0 + 32].to(cv5.DEV); n = len(x)
            w = (torch.rand(n, generator=g) * 2 - 1) * math.radians(15); m = 0.9 + 0.2 * torch.rand(n, generator=g)
            th = torch.zeros(n, 3, 4)
            D, H, W = x.shape[-3:]
            th[:, 0, 0] = torch.cos(w) / m; th[:, 0, 1] = -torch.sin(w) / m * (H / W)
            th[:, 1, 0] = torch.sin(w) / m * (W / H); th[:, 1, 1] = torch.cos(w) / m
            th[:, 2, 2] = 1.0
            gr = F.affine_grid(th.to(cv5.DEV), x.shape, align_corners=False)
            hirn = (F.grid_sample((x[:, :1] != 0).float(), gr, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5)
            x = F.grid_sample(x, gr, mode="bilinear", padding_mode="zeros", align_corners=False)
            y = F.grid_sample(y, gr, mode="nearest", padding_mode="zeros", align_corners=False)
            b = x[:, :1]
            anz = hirn.flatten(1).sum(1).clamp_min(1).view(n, 1, 1, 1, 1)
            mit = (b * hirn).flatten(1).sum(1).view(n, 1, 1, 1, 1) / anz
            std = (((b - mit) ** 2 * hirn).flatten(1).sum(1).view(n, 1, 1, 1, 1) / anz).sqrt().clamp_min(1e-6)
            r = lambda: torch.rand(n, 1, 1, 1, 1, generator=g).to(cv5.DEV)
            b = (b - mit) * (0.9 + 0.2 * r()) + mit + (r() * 0.2 - 0.1) * std
            b = b + torch.randn(b.shape, generator=g).to(cv5.DEV) * (0.05 * r()) * std
            b = torch.where(hirn, b, torch.zeros_like(b))
            x = torch.cat([b, x[:, 1:] * hirn], 1)
            aus_x.append(x.cpu()); aus_y.append(y.cpu())
        return torch.cat(aus_x), torch.cat(aus_y)
    return neu


def mit_intensitaets_augmentierung(stichprobe):
    """ONLY intensity, NO geometry (21 Sep 2026): nothing is interpolated, a bleed of a few voxels stays sharp. Per patch, each with
    probability 0.5, only image channel 0 and only in the brain: gamma 0.7-1.5 on the inverted scale, smooth multiplicative field (3x3x3 grid,
    exp(N(0, 0.15)), trilinear) like a residual bias field, contrast x(0.85-1.15) around the patch's brain mean, brightness +-0.1 std, noise up to 0.05 std.
    The FRST channel stays unchanged (it is normalized to relative contrast). Goal: scanner and scale differences between cohorts."""
    import torch.nn.functional as F
    def neu(train_ids, rng, P):
        xs, ys = stichprobe(train_ids, rng, P)
        g = torch.Generator(device="cpu").manual_seed(int(rng.randint(2 ** 31 - 1)))
        aus = []
        for b0 in range(0, len(xs), 32):
            x = xs[b0:b0 + 32].to(cv5.DEV); n = len(x)
            r = lambda: torch.rand(n, 1, 1, 1, 1, generator=g).to(cv5.DEV)
            an = lambda: (torch.rand(n, 1, 1, 1, 1, generator=g) < 0.5).float().to(cv5.DEV)
            b = x[:, :1]; hirn = b != 0
            gamma = 1 + an() * ((0.7 + 0.8 * r()) - 1)
            b = torch.where(hirn, b.clamp_min(1e-6) ** gamma, b)
            feld = F.interpolate(torch.exp(0.15 * torch.randn(n, 1, 3, 3, 3, generator=g)).to(cv5.DEV), size=b.shape[-3:], mode="trilinear", align_corners=True)
            b = b * (1 + an() * (feld - 1))
            anz = hirn.flatten(1).sum(1).clamp_min(1).view(n, 1, 1, 1, 1)
            mit = (b * hirn).flatten(1).sum(1).view(n, 1, 1, 1, 1) / anz
            std = (((b - mit) ** 2 * hirn).flatten(1).sum(1).view(n, 1, 1, 1, 1) / anz).sqrt().clamp_min(1e-6)
            b = (b - mit) * (1 + an() * (0.85 + 0.3 * r() - 1)) + mit + an() * (r() * 0.2 - 0.1) * std
            b = b + an() * torch.randn(b.shape, generator=g).to(cv5.DEV) * (0.05 * r()) * std
            b = torch.where(hirn, b.clamp(1e-6, 1.0), torch.zeros_like(b))
            aus.append(torch.cat([b, x[:, 1:]], 1).cpu())
        return torch.cat(aus), ys
    return neu


SCHICHT_WAHRSCHEINLICHKEIT = 0.35


def mit_schicht_augmentierung(stichprobe):
    """Thick-slice simulation (21 Sep 2026): with probability 0.35 a patch is treated as if it had been acquired with k = 2..5 mm slice thickness
    and recomputed to 1 mm: average over k slices, then linear interpolation in z (image and FRST). The label follows the acquisition: a bleed that touches a
    thick slice counts across the whole slice (maximum over k, nearest neighbor back). Background stays 0. Goal: the thin-slice cases
    (VALDO cohort 2) supply training examples for the appearance of the thick-slice ones (cohorts 1 / 3 and the second cohort), where we are weakest."""
    import torch.nn.functional as F
    def neu(train_ids, rng, P):
        xs, ys = stichprobe(train_ids, rng, P)
        D = xs.shape[2]
        for j in range(len(xs)):
            if rng.rand() >= SCHICHT_WAHRSCHEINLICHKEIT:
                continue
            k = int(rng.randint(2, 6))
            x = xs[j:j + 1].to(cv5.DEV); y = ys[j:j + 1].to(cv5.DEV)
            hirn = (x[:, :1] != 0).float()
            dick = F.avg_pool3d(x, (k, 1, 1), stride=(k, 1, 1), ceil_mode=True, count_include_pad=False)
            x2 = F.interpolate(dick, size=x.shape[-3:], mode="trilinear", align_corners=False) * hirn
            y2 = F.interpolate(F.max_pool3d(y, (k, 1, 1), stride=(k, 1, 1), ceil_mode=True), size=y.shape[-3:], mode="nearest") * hirn
            xs[j], ys[j] = x2[0].cpu(), y2[0].cpu()
        return xs, ys
    return neu


def falte_ordner(basis, k):
    """Like cv5.falte_ordner: only meta.ok counts as done, incomplete folds are set aside."""
    outd = f"{basis}/fold{k}"
    mp = f"{outd}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("ok"):
        return outd, True
    if os.path.isdir(outd) and os.listdir(outd):
        ab = f"{outd}_abgebrochen-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(outd, ab)
        print(f"incomplete fold set aside: {ab}", flush=True)
    os.makedirs(outd, exist_ok=True)
    return outd, False


def eine_falte(netz, k, outd, **kw):
    """train_falte with one OOM catch: batch 4 -> 2 (with InstanceNorm this changes the
    normalization not at all, only the number of steps). Recorded in meta.json."""
    abw = None
    _INIT["fold"] = k
    try:
        m = cv5.train_falte(netz, k, outd, **kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        abw = 2
        print(f"[{netz} f{k}] CUDA OOM at batch {cv5.REZEPT['bs']} -> retry with batch 2", flush=True)
        cv5.KONFS[netz]["bs"] = 2
        shutil.rmtree(outd); os.makedirs(outd)
        m = cv5.train_falte(netz, k, outd, **kw)
    m.update(parameter=_LETZTES.get("parameter"), vortrainiert=_LETZTES.get("vortrainiert"),
             bs_abweichend=abw, seed=cv5.SEED)
    json.dump(m, open(f"{outd}/meta.json", "w"), indent=1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--netz", nargs="*", default=list(NETZE))
    ap.add_argument("--falten", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kanaele", choices=["t2s_frst", "t2s"], default="t2s_frst",
                    help="t2s = control without the FRST channel (folder gets the suffix -t2s)")
    ap.add_argument("--epochen", type=int, default=None,
                    help="Recipe 2 (19 Sep): fixed training duration instead of early stopping -- max_ep = min_ep = N, "
                         "inner fold evaluated every N/12 epochs; folder suffix -e<N>")
    ap.add_argument("--anteil-laesion", type=float, default=None,
                    help="Fraction of patches WITH a bleed (recipe 0.60 = 1.5:1). The negatives keep the "
                         "ratio brain:arbitrary = 25:15. 0.50 = 1:1, 0.333 = 1:2; folder suffix -l<percent>")
    ap.add_argument("--landmarken", default="dev/landmarken_valdo.json",
                    help="Reference file for --norm landmarken; MUST be the same for training and prediction")
    ap.add_argument("--norm", choices=["max", "z", "p995", "landmarken"], default="max",
                    help="z = z-normalization of the T2* channel per case in the brain (like nnU-Net), background stays 0; "
                         "max = as delivered (1 - image/maximum). Folder suffix -nz")
    ap.add_argument("--aug", action="store_true",
                    help="extra augmentation (GPU): in-plane rotation +-15 degrees, scale 0.9-1.1, "
                         "brightness/contrast +-10 %%, noise; flips stay. Folder suffix -aug")
    ap.add_argument("--aug-intensitaet", action="store_true",
                    help="only intensity augmentation (gamma, smooth field, contrast, brightness, noise), no interpolation; folder suffix -augi")
    ap.add_argument("--aug-schicht", action="store_true",
                    help="thick-slice simulation in z (k = 2..5 mm, probability 0.35), label follows; folder suffix -augs")
    ap.add_argument("--hp", nargs="*", default=[], metavar="SCHLUESSEL=WERT",
                    help="Tournament 2: override recipe values (lr, weight_decay, bs, n_patches, anteil_laesion, anteil_hirn); folder suffix -hp<...>")
    ap.add_argument("--verlust", choices=["bce_dice", "blob", "zentrum", "fnrw", "focal_tversky", "bce_dice_ohne_ds"], default="bce_dice",
                    help="blob = BCE+Dice plus one term per bleed (blob loss, Kofler 2023); zentrum = BCE+Dice plus center map as "
                         "auxiliary output (CenSynCMB 2026), only with --netz a12-anisozentrum; fnrw = patches weighted by the size and current "
                         "network response of their bleeds (CenSynCMB); folder suffix -v<name>")
    ap.add_argument("--gewebe", nargs="?", const="dev/synthseg/{id}_synthseg.nii.gz", default=None, metavar="MUSTER",
                    help="five one-hot tissue channels from the SynthSeg map as extra input (CSF, cortex, white matter, deep nuclei, "
                         "infratentorial); pattern with {id} or {sid}, default VALDO; folder suffix -gew")
    ap.add_argument("--letzter-stand", action="store_true",
                    help="Counter-check 21 Sep: the test prediction comes from the LAST intermediate state (a truly fixed epoch count) instead of the best "
                         "intermediate state of the inner fold; threshold / min size still from the inner fold. Folder suffix -ls")
    ap.add_argument("--synthese", default=None, metavar="GITTER",
                    help="grid with synthetic bleeds/mimics (cmb.synthesis.build_synthetic_grid); training cases only; folder suffix -syn")
    ap.add_argument("--synthese-anteil", type=float, default=None, metavar="ANTEIL",
                    help="with --synthese: second version -- real sampling unchanged, ANTEIL of the bleed-free patches replaced by synthetic bleeds; folder suffix -syn<percent>")
    ap.add_argument("--zusatz", nargs="+", default=None, metavar="NAME",
                    help="continuous extra channels <id>_<NAME>.nii.gz from the grid folder (e.g. t1 t2; code/zusatzkanaele_bauen.py); folder suffix -z<names>")
    ap.add_argument("--harte-negative", action="store_true",
                    help="from epoch 11 every 10 epochs mine the network's own false positives on the training cases and put 60 %% of the "
                         "bleed-free patches there; folder suffix -hn")
    ap.add_argument("--lr", type=float, default=None, help="learning rate instead of 3e-4; folder suffix -lr<value>")
    ap.add_argument("--maske", choices=["valdo", "synthseg"], default="valdo",
                    help="synthseg = set image and FRST to 0 outside the SynthSeg brain (dilated by 2 voxels). "
                         "The delivered VALDO mask is 40 %% non-brain (eyes, skull base) in cohort 3, "
                         "16 %% in cohort 2 (measured 19 Sep). Folder suffix -ms")
    ap.add_argument("--manifest", nargs="*", default=None,
                    help="19 Sep transfer: one or more JSON lists [{id, fold, dir, cohort}, ...] instead of dev/cases.json + "
                         "dev/split.json. Cases from several cohorts are trained together in 5 folds (fold = field fold).")
    ap.add_argument("--name", default=None, help="folder suffix for --manifest/--init runs, e.g. mix-t2s")
    ap.add_argument("--init", default=None,
                    help="fine-tuning: starting weights per fold, placeholder {fold}, e.g. .../a03-aniso-e60-gd/fold{fold}/modell.pt")
    ap.add_argument("--basis", default=None, help="result folder instead of ergebnisse/turnier; private cohorts get a folder outside this tree here")
    ap.add_argument("--gitter", default=None,
                    help="different data grid instead of dev/gitter_c, e.g. dev/gitter_d (without vessel inpainting); "
                         "folder suffix -g<name>")
    ap.add_argument("--patch", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"),
                    help="patch size in voxels (z y x) instead of cv5.PATCH (38 62 94); folder suffix -p<Z>x<Y>x<X>")
    ap.add_argument("--bs", type=int, default=None, help="only --probe: batch (also for the tiles)")
    ap.add_argument("--vram-anteil", type=float, default=None,
                    help="only --probe: cap for the own VRAM fraction (protects the model server)")
    a = ap.parse_args()
    if a.liste:
        for n in NETZE:
            print(n)
        return
    for n in a.netz:
        assert n in NETZE, f"unknown network {n!r}; --liste"
    np.random.seed(a.seed)
    cv5.SEED = a.seed
    for n in a.netz:
        cv5.KONFS[n]["kanaele"] = a.kanaele
    zusatz = ("-t2s" if a.kanaele == "t2s" else "") + ("" if a.seed == 42 else f"-s{a.seed}")
    kw_lang = {}
    if a.epochen:
        kw_lang = dict(max_ep=a.epochen, min_ep=a.epochen, eval_jede=max(a.epochen // 12, 1))
        zusatz += f"-e{a.epochen}"
    if a.anteil_laesion is not None:
        assert 0.05 <= a.anteil_laesion <= 0.95
        for n in a.netz:
            cv5.KONFS[n]["anteil_laesion"] = a.anteil_laesion
            cv5.KONFS[n]["anteil_hirn"] = round((1 - a.anteil_laesion) * 25 / 40, 4)
        zusatz += f"-l{round(a.anteil_laesion * 100)}"
    if a.gitter:
        cv5.GITTER = a.gitter if os.path.isabs(a.gitter) else f"{ROOT}/{a.gitter}"
        assert os.path.isdir(cv5.GITTER), cv5.GITTER
        zusatz += "-g" + os.path.basename(cv5.GITTER).replace("gitter_", "")
    if a.norm == "z":
        zusatz += "-nz"
    if a.norm == "p995":
        zusatz += "-np995"
    if a.norm == "landmarken":
        zusatz += "-nlm"
    if a.aug:
        cv5.stichprobe = mit_augmentierung(cv5.stichprobe)
        zusatz += "-aug"
    if a.aug_schicht:
        cv5.stichprobe = mit_schicht_augmentierung(cv5.stichprobe)
        zusatz += "-augs"
    if a.aug_intensitaet:
        cv5.stichprobe = mit_intensitaets_augmentierung(cv5.stichprobe)
        zusatz += "-augi"
    if a.harte_negative:
        cv5.stichprobe = mit_harten_negativen(cv5.stichprobe)
        zusatz += "-hn"
    if a.synthese:
        g = a.synthese if os.path.isabs(a.synthese) else f"{ROOT}/{a.synthese}"
        assert os.path.isdir(g), g
        if a.synthese_anteil is not None:
            assert 0 < a.synthese_anteil <= 1
            cv5.stichprobe = mit_synthese_anteil(cv5.stichprobe, g, a.synthese_anteil)
            zusatz += f"-syn{round(a.synthese_anteil * 100)}" + ("-" + os.path.basename(g).replace("gitter_", "") if os.path.basename(g) != "gitter_s" else "")
        else:
            cv5.stichprobe = mit_synthese(cv5.stichprobe, g)
            zusatz += "-syn"
    if a.letzter_stand:
        assert a.epochen, "--letzter-stand needs --epochen (fixed plan)"
        cv5.LETZTER_STAND = True
        zusatz += "-ls"
    if a.lr:
        for n in a.netz:
            cv5.KONFS[n]["lr"] = a.lr
        zusatz += f"-lr{a.lr:g}"
    if ("a12-anisozentrum" in a.netz) != (a.verlust == "zentrum"):
        raise SystemExit("--netz a12-anisozentrum and --verlust zentrum belong together (and only together)")
    if a.verlust != "bce_dice":
        for n in a.netz:
            cv5.KONFS[n]["loss"] = "bce_dice" if a.verlust == "bce_dice_ohne_ds" else a.verlust
        zusatz += "-v" + a.verlust
    for hp in a.hp:                                       # Tournament 2: one recipe value per run
        k, v = hp.split("=", 1)
        assert k in ("lr", "weight_decay", "bs", "n_patches", "anteil_laesion", "anteil_hirn"), k
        val = int(v) if k in ("bs", "n_patches") else float(v)
        for n in a.netz:
            cv5.KONFS[n][k] = val
        zusatz += f"-hp{k.replace('_', '')}{v}"
    if a.maske == "synthseg":
        cv5.lade_daten = mit_synthseg_maske(cv5.lade_daten)     # BEFORE the z-normalization: that then computes only in the brain
        zusatz += "-ms"
    if a.manifest:
        if a.maske != "valdo" or a.gitter:
            raise SystemExit("--manifest does not (yet) work together with --maske/--gitter: the manifest loader replaces their loader")
        cv5.lade_daten = mit_manifest(a.manifest)
    # The normalization is wrapped in LAST, so that it wraps the loader actually in use -- whether grid
    # or manifest. Before, it was wrapped earlier and got silently overwritten by the manifest loader; that's why
    # "--manifest + --norm" was forbidden. Exactly that combination is needed for mixing with harmonization (23 Sep 2026).
    if a.norm == "z":
        cv5.lade_daten = mit_z_normierung(cv5.lade_daten)
    if a.norm == "p995":
        cv5.lade_daten = mit_perzentil_normierung(cv5.lade_daten)
    if a.norm == "landmarken":
        cv5.lade_daten = mit_landmarken_normierung(cv5.lade_daten, a.landmarken if os.path.isabs(a.landmarken) else f"{ROOT}/{a.landmarken}")
    if a.gewebe:                                          # AFTER choosing the loader (grid or manifest): attaches the map to every case
        if a.harte_negative or a.aug or a.aug_schicht or a.aug_intensitaet or a.probe:
            raise SystemExit("--gewebe does not (yet) work together with --harte-negative / --aug / --probe: their sampling does not know the extra channels")
        _GEWEBE["an"] = True
        cv5.lade_daten = mit_gewebekanal(cv5.lade_daten, a.gewebe)
        cv5.stichprobe, cv5.proba_fall = stichprobe_gewebe, proba_fall_gewebe
        zusatz += "-gew"
    if a.zusatz:                                          # AFTER choosing the loader, like --gewebe
        if a.harte_negative or a.aug or a.aug_schicht or a.aug_intensitaet or a.probe or a.manifest or a.synthese:
            raise SystemExit("--zusatz does not (yet) work together with --harte-negative / --aug / --probe / --manifest / --synthese")
        _ZUSATZ["namen"] = list(a.zusatz)
        cv5.lade_daten = mit_zusatzkanaelen(cv5.lade_daten, _ZUSATZ["namen"])
        cv5.stichprobe, cv5.proba_fall = stichprobe_gewebe, proba_fall_gewebe
        zusatz += "-z" + "".join(a.zusatz)
    if a.init:
        _INIT["muster"] = a.init
    if a.name:
        zusatz += "-" + a.name
    if a.patch:
        cv5.PATCH = tuple(a.patch)
        zusatz += "-p" + "x".join(map(str, a.patch))

    if a.probe:
        if a.vram_anteil and cv5.DEV == "cuda":
            torch.cuda.set_per_process_memory_fraction(a.vram_anteil)
        if a.bs:
            cv5.proba_fall = functools.partial(cv5.proba_fall, bs=a.bs)
        folds = json.load(open(f"{ROOT}/dev/split.json"))["folds"]
        cv5.lade_daten(ids=set(folds[0][:2] + folds[1][:2] + folds[2][:3]))   # 7 cases are enough
        aus = {}
        for n in a.netz:
            if a.bs:
                cv5.KONFS[n]["bs"] = a.bs
            outd = f"{ROOT}/ergebnisse/turnier-probe/{n}{zusatz}/fold0"
            if os.path.isdir(outd):
                shutil.rmtree(outd)
            os.makedirs(outd)
            t0 = time.time()
            try:
                m = cv5.train_falte(n, 0, outd, max_ep=2, min_ep=2, eval_jede=1, n_patches=40)
                aus[n] = dict(ok=True, dauer_s=round(time.time() - t0, 1), vram_gib=m["vram_gib"],
                              parameter=_LETZTES.get("parameter"), vortrainiert=_LETZTES.get("vortrainiert"))
            except Exception as e:
                if cv5.DEV == "cuda":
                    torch.cuda.empty_cache()
                aus[n] = dict(ok=False, fehler=f"{type(e).__name__}: {e}"[:300])
            print("PROBE", n, json.dumps(aus[n], ensure_ascii=False), flush=True)
        alt = {}
        pp = f"{ROOT}/dev/turnier_probe.json"
        if os.path.exists(pp):
            alt = json.load(open(pp))
        alt.update(aus)
        json.dump(alt, open(pp, "w"), indent=1)
        sys.exit(0 if all(v["ok"] for v in aus.values()) else 1)

    cv5.lade_daten()
    status_p = f"{a.basis}/turnier_status.json" if a.basis else f"{ROOT}/dev/turnier_status.json"
    fehler = 0
    for n in a.netz:
        name = n + zusatz
        basis = f"{a.basis or ROOT + '/ergebnisse/turnier'}/{name}"
        for k in a.falten:
            outd, fertig = falte_ordner(basis, k)
            if fertig:
                print(f"[{name} f{k}] done -- skipped", flush=True); continue
            t0 = time.time()
            try:
                m = eine_falte(n, k, outd, zeit_limit=FOLD_ZEIT_S, **kw_lang)
                eintrag = dict(ok=True, f1=m["f1_testfalte"], dauer_s=m["dauer_s"], vram_gib=m["vram_gib"],
                               bs_abweichend=m["bs_abweichend"])
            except Exception as e:
                fehler += 1
                if cv5.DEV == "cuda":
                    torch.cuda.empty_cache()
                eintrag = dict(ok=False, fehler=f"{type(e).__name__}: {e}"[:300], dauer_s=round(time.time() - t0, 1))
                print(f"[{name} f{k}] ERROR: {eintrag['fehler']}", flush=True)
            # Re-read status: several network runs write into the same file one after another
            status = json.load(open(status_p)) if os.path.exists(status_p) else {}
            status[f"{name}/fold{k}"] = eintrag
            json.dump(status, open(status_p, "w"), indent=1)
    print(f"TOURNAMENT {' '.join(a.netz)}: {fehler} fold(s) with an error.", flush=True)
    sys.exit(1 if fehler else 0)


if __name__ == "__main__":
    main()
