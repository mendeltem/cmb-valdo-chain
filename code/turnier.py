#!/usr/bin/env python3
"""Architektur-Turnier auf VALDO T2* (Claude 18.09.2026, Nutzerauftrag).

Setzt die Netze aus code/netze.py in den UNVERAENDERTEN cv5-Ablauf ein: dieselben fuenf Falten
(dev/split.json), dasselbe Gitter C, dieselbe Stichprobe, dieselbe innere Falte fuer Epoche und
Schwelle, derselbe Klumpen-Waechter, dieselbe Metrik (metric.hits, 26er-Komponenten). Je Netz
aendert sich NUR die Architektur -- Rezept = cv5 r3 (T2* + FRST, InstanceNorm, BCE+Dice,
AdamW 3e-4 cosine, 800 Patches/Epoche, Stapel 4). Das Validierungsset wird nicht gelesen.

  python turnier.py --liste
  python turnier.py --probe [--netz a04-2p5d ...] [--bs 1] [--vram-anteil 0.06]
        Mini-Falte je Netz auf der GPU (2 Epochen, 2 Auswertungen, 40 Patches): jeder Pfad
        einmal, auch bfloat16-Autocast. Schreibt nach ergebnisse/turnier-probe/, nie ins Turnier.
  python turnier.py --netz a01-attunet [--falten 0 1 2 3 4] [--seed 42] [--kanaele t2s]
        Ergebnisse: ergebnisse/turnier/<netz>[-t2s][-s<seed>]/fold<k>/ (Aufbau wie cv5),
        Status: dev/turnier_status.json. Wiederaufnehmbar: fertige Falten werden uebersprungen.
Rueckgabewert: 0 alles ok, 1 mindestens eine Falte mit Fehler.

Abweichungen vom Rezept, die nur hier vorkommen koennen, stehen in meta.json und im Status:
  bs_abweichend  CUDA-OOM bei Stapel 4 -> Wiederholung mit Stapel 2 (nur grosse Netze)
  vortrainiert   Zahl der geladenen Encoder-Tensoren (nur a08-swin)
"""
import os, sys, json, time, shutil, argparse, functools
import numpy as np
import torch

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
import cv5
from netze import NETZE, parameterzahl

FOLD_ZEIT_S = 6 * 3600          # Zeitzaun je Falte; cv5 hat 3 h, Swin mit Checkpointing ist langsamer
_LETZTES = {}                   # das zuletzt gebaute Modell (fuer Parameterzahl/vortrainiert in meta)


def loss_bce_dice_ds(lg, ziel):
    """Wie cv5.loss_bce_dice. Liefert das Netz Deep-Supervision-Koepfe (B, K, 1, D, H, W), werden
    sie 1, 1/2, 1/4, ... gewichtet (nnU-Net). In eval() kommt immer nur der Hauptkopf."""
    if lg.dim() == 5:
        return cv5.loss_bce_dice(lg, ziel)
    gew = [0.5 ** i for i in range(lg.shape[1])]
    return sum(g * cv5.loss_bce_dice(lg[:, i], ziel) for i, g in enumerate(gew)) / sum(gew)


BLOB_GEWICHT = 0.5             # global + 0.5 * blob  ==  Koflers alpha=2, beta=1 bis auf den Faktor 2


def loss_blob(lg, ziel):
    """blob loss (Kofler et al., arXiv:2205.08209; Tagung am 23.09.2026 an der Quelle bestaetigt: IPMI 2023, Springer LNCS) auf BCE+Dice: zum globalen Verlust kommt je
    Blutung ein eigener Term, gerechnet auf dem Ausschnitt OHNE die Voxel der anderen Blutungen. So zaehlt
    eine 4-Voxel-Blutung so viel wie eine mit 80 Voxeln; im globalen Dice geht sie unter. Komponenten mit
    26er-Nachbarschaft wie im Mass. Ausschnitte ohne Blutung tragen nur den globalen Term. Bei
    Deep-Supervision-Koepfen wirkt der Blob-Term nur auf den Hauptkopf. Gerechnet in float32 (kleine Summen)."""
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
            m = ((marken == j) | (marken == 0)).float()      # diese Blutung + Hintergrund, die anderen ausgeblendet
            bce = (bce_voxel * m).sum() / m.sum()
            dice = 1 - (2 * (p * zb * m).sum() + 1.0) / ((p * m).sum() + (zb * m).sum() + 1.0)
            terme.append(0.5 * bce + 0.5 * dice)
    if not terme:
        return basis
    return basis + BLOB_GEWICHT * torch.stack(terme).mean()


ZENTRUM_SIGMA_MM = 2.0          # Gauss-Breite der Zentrumskarte in mm (CenSynCMB: 2 Voxel bei 1 mm isotrop)
ZENTRUM_GAMMA, ZENTRUM_EPS = 2.5, 0.05   # Gewicht (C + eps)^gamma; CenSynCMB nennt eps = 1e-3, siehe loss_zentrum
ZENTRUM_GEWICHT = 1.0
GITTER_MM = (1.0, 0.5, 0.5)     # Voxelgroesse der Trainingsgitter in (z, y, x)


def baue_aniso_mit_zentrum(nch):
    """a03-aniso mit zweitem Kopf fuer eine ZENTRUMSKARTE (He et al. 2026, CenSynCMB, arXiv 2607.05325: groesster
    Einzelbeitrag ihrer Ablation, +4.5 F1-Punkte). Der Kopf sitzt auf denselben Decoder-Merkmalen wie der
    Segmentierkopf (Vorwaerts-Haken auf `aus`). Im Training liefert das Netz (B, 2, D, H, W) = Segmentierung +
    Zentrum, in eval() NUR die Segmentierung -- Auswertung, TTA, Ensemble und zweite Stufe bleiben unveraendert."""
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

    return Polster(_AnisoMitZentrum(nch), (8, 16, 16))      # dieselbe Polsterung wie a03-aniso


def zentrumskarte(ziel):
    """(B, 1, D, H, W) Label -> Gauss-Hoecker je Blutung am Schwerpunkt, C = max_k exp(-d_k^2 / (2 sigma^2)), d in mm."""
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
    """BCE+Dice auf dem Segmentierkopf + fokaler MSE auf der Zentrumskarte (CenSynCMB). Abweichungen vom Paper, bewusst:
    (1) genormt auf die Gewichtssumme statt auf die Voxelzahl und festes Gewicht 1 statt gelernter Unsicherheit -- unser
    Verlust sieht die Netzparameter nicht; (2) eps = 0.05 statt 1e-3: mit 1e-3 zaehlt der Hintergrund praktisch nicht, mit
    0.05 lernt der Kopf eine brauchbare Detektionskarte (spaeter als zweite Meinung nutzbar); (3) sigma in mm, weil unser
    Gitter 0.5 x 0.5 x 1 mm hat. Ausschnitte ohne Blutung: Zielkarte 0, der Kopf lernt dort 'kein Zentrum'."""
    if lg.dim() != 5 or lg.shape[1] != 2:
        raise RuntimeError("--verlust zentrum braucht das Netz a12-anisozentrum (zwei Ausgabekanaele im Training)")
    basis = cv5.loss_bce_dice(lg[:, :1], ziel)
    karte = zentrumskarte(ziel)
    gewicht = (karte + ZENTRUM_EPS) ** ZENTRUM_GAMMA
    fehler = (torch.sigmoid(lg[:, 1:2].float()) - karte) ** 2
    return basis + ZENTRUM_GEWICHT * (gewicht * fehler).sum() / gewicht.sum()


FNRW_GAMMA, FNRW_RHO, FNRW_GRENZEN = 0.5, 1.5, (0.5, 5.0)     # Werte aus CenSynCMB (He et al. 2026)
_FNRW = {"mittlere_groesse": None}


def loss_fnrw(lg, ziel):
    """BCE+Dice JE AUSSCHNITT, gewichtet nach den Blutungen darin (CenSynCMB: 'false-negative-driven reweighting'):
    w_k = (mittlere Groesse / Groesse_k)^0.5 * (1 + 1.5 * (1 - r_k)), r_k = hoechste aktuelle Netzantwort auf der Blutung.
    Kleine und gerade schwach erkannte Blutungen zaehlen mehr; Ausschnitte ohne Blutung behalten Gewicht 1. Das Paper sagt
    nicht, wie mehrere Blutungen in einem Ausschnitt zusammengehen -- hier das MAXIMUM (die schwierigste bestimmt das
    Gewicht), begrenzt auf [0.5, 5]. Bei lauter Gewichten 1 ist der Wert exakt cv5.loss_bce_dice."""
    import torch.nn.functional as F
    from scipy import ndimage
    if lg.dim() != 5 or lg.shape[1] != 1:
        raise RuntimeError("--verlust fnrw erwartet ein Netz mit einem Ausgabekanal ohne Deep Supervision")
    if _FNRW["mittlere_groesse"] is None:               # Skalenkonstante aus den geladenen Faellen (keine Label-Information ueber Testfaelle noetig)
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


HN_START, HN_JEDE, HN_ANTEIL = 10, 10, 0.6       # ab Epoche 11 alle 10 Epochen schuerfen; 60 % der blutungsfreien Ausschnitte ersetzen
_HN = {"aufrufe": 0, "zentren": []}


def fehlalarm_zentren(modell, train_ids, nch):
    """Das AKTUELLE Netz sagt seine eigenen Trainingsfaelle vorher (feste Nachbearbeitung 0.3 / 8 Voxel); Schwerpunkte der
    Komponenten, die keine Referenz beruehren, werden Stichprobenzentren. Streng innerhalb der Falte: kein fremder Lauf,
    kein Testfall."""
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
    """Online hard negative mining in der Stichprobe (verwandt mit Dou et al. 2016, wo aber die ZWEITE Stufe auf den Fehlalarmen der ersten lernt): ein Teil der blutungsfreien Ausschnitte
    wird durch Ausschnitte um die eigenen Fehlalarme des Netzes ersetzt. Die Referenz im Ausschnitt bleibt die echte --
    liegt neben dem Fehlalarm eine Blutung, steht sie korrekt im Label."""
    def neu(train_ids, rng, P):
        _HN["aufrufe"] += 1
        ep = _HN["aufrufe"]
        modell = _LETZTES.get("modell")
        nch = 1 if P["kanaele"] == "t2s" else 2
        if modell is not None and ep > HN_START and (ep - HN_START - 1) % HN_JEDE == 0:
            t0 = time.time()
            _HN["zentren"] = fehlalarm_zentren(modell, train_ids, nch)
            modell.train()                                    # proba_fall schaltet auf eval()
            print(f"[harte Negative] Epoche {ep}: {len(_HN['zentren'])} Fehlalarm-Zentren in "
                  f"{len({s for s, _ in _HN['zentren']})} von {len(train_ids)} Trainingsfaellen ({time.time() - t0:.0f} s)", flush=True)
        xs, ys = stichprobe(train_ids, rng, P)
        if _HN["zentren"]:
            leer = [j for j in range(len(ys)) if float(ys[j].sum()) == 0.0]
            for j in leer[:int(len(leer) * HN_ANTEIL)]:       # Reihenfolge ist schon gemischt (Permutation in cv5.stichprobe)
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


# ---------------------------------------------------------------- Kuenstliche Blutungen und Verwechsler (28.09.2026, ARBEIT.md 5ae)
# cmb/synthesis/build_synthetic_grid.py schreibt ein Gitter, in dem jeder Fall zusaetzliche kuenstliche Blutungen (im Label)
# und Gefaess-Verwechsler (nur im Bild) traegt, FRST neu gerechnet. Hier werden NUR die Trainingsfaelle der Falte gegen diese
# Fassung getauscht, waehrend die Stichprobe gezogen wird; Test- und innere Wahlfalte bleiben echt, und nach dem Ziehen
# stehen die echten Daten wieder in cv5.FA -- proba_fall (Wahl des Zwischenstands, Testkarte) sieht nie ein kuenstliches Voxel.
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
            assert img.shape == echt["img"].shape, f"{sid}: Synthesegitter passt nicht zum Trainingsgitter"
            assert (lab >= echt["lab"]).all(), f"{sid}: Synthese-Label enthaelt das echte nicht"
            ll, n = ndimage.label(lab, structure=cv5.N26)
            syn = dreh(np.asarray(nib.load(f"{d}/{sid}_synth.nii.gz").dataobj) == 1)         # 1 = kuenstliche Blutung
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
        if _SYN["ids"] != ids:                              # neue Falte: Zwischenspeicher der alten Falte freigeben
            _SYN["recs"] = {k: v for k, v in _SYN["recs"].items() if k in ids}
            _SYN["ids"] = ids
            print(f"[Synthese] {len(ids)} Trainingsfaelle aus {gitter}", flush=True)
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
    """Zweite Fassung (ARBEIT.md 5ae-1b, 28.09.2026): das Budget der ECHTEN Laesionen bleibt exakt wie in der Basis -- die
    Stichprobe wird unveraendert gezogen; danach wird der Anteil `anteil` der Ausschnitte OHNE Laesion durch Ausschnitte um
    kuenstliche Blutungen aus dem Synthesegitter ersetzt (Bild, FRST und Label der kuenstlichen Fassung, also mit den
    kuenstlichen Blutungen im Label). Verwechsler gibt es in dieser Fassung nicht. Test- und Wahlfalte bleiben echt."""
    _SYN["gitter"] = gitter

    def neu(train_ids, rng, P):
        xs, ys = stichprobe(train_ids, rng, P)
        ids = tuple(sorted(train_ids))
        if _SYN["ids"] != ids:
            _SYN["recs"] = {k: v for k, v in _SYN["recs"].items() if k in ids}
            _SYN["ids"] = ids
            print(f"[Synthese-Anteil {anteil}] {len(ids)} Trainingsfaelle aus {gitter}", flush=True)
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


# ---------------------------------------------------------------- Gewebekarte als Eingabekanaele (21.09.2026)
# SynthSeg-Label -> Klasse 1..5. Bisher nutzen wir die Karte nur NACH der Vorhersage (Liquor-Regel, +0.06 auf VALDO). Als
# Eingabe kann das Netz selbst lernen, wo Verwechsler sitzen (Sulci, Ventrikelrand, Stammganglien) -- andere Information
# als das T2*-Bild, ohne weitere Sequenz bei der Zielkohorte (dort rechnet SynthSeg --robust direkt auf T2*).
GEWEBE_KLASSEN = {}
for _l in (4, 5, 14, 15, 24, 43, 44):                          GEWEBE_KLASSEN[_l] = 1      # Liquor
for _l in (3, 42, 17, 18, 53, 54):                             GEWEBE_KLASSEN[_l] = 2      # Rinde + mesiotemporal
for _l in (2, 41):                                             GEWEBE_KLASSEN[_l] = 3      # Grosshirn-Mark
for _l in (10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60):    GEWEBE_KLASSEN[_l] = 4      # tiefe Kerne
for _l in (7, 8, 16, 46, 47):                                  GEWEBE_KLASSEN[_l] = 5      # Kleinhirn + Hirnstamm
N_GEWEBE = 5
_GEWEBE = {"an": False}


def mit_gewebekanal(lade, muster):
    """Haengt an jeden Fall `gew` (uint8, 0..5, (z, y, x)). Karten auf anderem Gitter (zweite Kohorte: 1 mm) werden mit
    naechstem Nachbarn auf das Trainingsgitter gelegt (dieselbe Funktion wie in cmb.analysis)."""
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
            assert os.path.exists(pfad), f"Gewebekarte fehlt: {pfad}"
            wie = nib.Nifti1Image(np.zeros(r["img"].shape[::-1], np.uint8), r["affine"])      # Gitter des Falls in (x, y, z)
            seg = load_synthseg(pfad, wie)
            gew = tabelle[np.clip(seg, 0, 255)].transpose(2, 1, 0)
            assert gew.shape == r["img"].shape, sid
            r["gew"] = np.ascontiguousarray(gew)
            anteile += np.bincount(r["gew"][r["img"] != 0].ravel(), minlength=N_GEWEBE + 1) / max(int((r["img"] != 0).sum()), 1)
        anteile /= max(len(cv5.FA["recs"]), 1)
        print("Gewebekanaele: Anteil am Hirn (ohne / Liquor / Rinde / Mark / tief / infratentoriell): "
              + " / ".join(f"{v:.2f}" for v in anteile), flush=True)
    return neu


_ZUSATZ = {"namen": []}          # Arm 8 (29.09.2026): stetige Zusatzkanaele <id>_<name>.nii.gz aus dem Gitterordner, z. B. t1 t2


def mit_zusatzkanaelen(lade, namen):
    """Haengt an jeden Fall r["zus"] = [Karte je Name] ((z, y, x), float32, dieselbe Form wie das Bild). Die Karten liegen
    fertig normiert im Gitterordner des Falls (code/zusatzkanaele_bauen.py)."""
    import nibabel as nib

    def neu(*a, **kw):
        lade(*a, **kw)
        dreh = lambda arr: np.ascontiguousarray(arr.transpose(2, 1, 0))
        for sid, r in cv5.FA["recs"].items():
            r["zus"] = []
            for name in namen:
                pfad = f"{cv5.GITTER}/{sid}/{sid}_{name}.nii.gz"
                assert os.path.exists(pfad), f"Zusatzkanal fehlt: {pfad}"
                k = dreh(np.asarray(nib.load(pfad).dataobj, dtype=np.float32))
                assert k.shape == r["img"].shape, (sid, name, k.shape, r["img"].shape)
                r["zus"].append(k)
        print(f"Zusatzkanaele {namen}: {len(cv5.FA['recs'])} Faelle", flush=True)
    return neu


def _n_extra():
    return (N_GEWEBE if _GEWEBE["an"] else 0) + len(_ZUSATZ["namen"])


def _kanaele(r, st, nch):
    """Bild (+ FRST) (+ fuenf One-hot-Gewebekanaele) (+ stetige Zusatzkanaele) fuer den Ausschnitt ab Start st."""
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
    """WORTGLEICH cv5.stichprobe (gleiche Reihenfolge der Zufallszahlen -> bei gleichem Startwert dieselben Ausschnitte wie
    die Basis), nur mit den zusaetzlichen Kanaelen."""
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
    """Wie cv5.proba_fall (halber Stride, Ueberlapp-Mittel, leere Kacheln uebersprungen), mit den zusaetzlichen Kanaelen."""
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
        print(f"Startgewichte geladen: {pfad}", flush=True)
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
    """Ersetzt cv5.lade_daten: Faelle aus Manifest-Listen (mehrere Kohorten moeglich). train_falte liest die Falten aus
    f"{cv5.ROOT}/dev/split.json" -- dafuer wird cv5.ROOT auf einen Hilfsordner mit einer passenden split.json umgebogen
    (cv5.ROOT wird in train_falte NUR dafuer benutzt; die Ergebnisordner bestimmt turnier.py selbst)."""
    import hashlib
    import nibabel as nib
    from scipy import ndimage
    eintraege = [e for p in pfade for e in json.load(open(p))]
    assert len({e["id"] for e in eintraege}) == len(eintraege), "Fall-IDs muessen ueber alle Manifeste eindeutig sein"
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
        print(f"Daten aus Manifest: {len(recs)} Faelle {je}, {sum(len(r['komp']) for r in recs.values())} CMB-Komponenten", flush=True)
    return lade


def mit_synthseg_maske(lade):
    """Setzt Bild und FRST ausserhalb des SynthSeg-Hirns (Label > 0, um 2 Voxel geweitet) auf 0. Die Label
    bleiben unangetastet; liegt eine Referenz-CMB ausserhalb, wird das gezaehlt und gemeldet."""
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
        print(f"SynthSeg-Maske: im Mittel {100 * np.mean(weg):.0f} % der alten Maske entfernt; "
              f"Referenz-CMB ganz ausserhalb: {cmb_aussen}", flush=True)
    return neu


def mit_perzentil_normierung(lade, q=99.5):
    """Robuste Fassung der Original-Skala. Geliefert ist Bild = 1 - T2*/Maximum; ein einzelner heller Ausreisser
    staucht dann das ganze Hirn zusammen. Hier: T2*/Maximum = 1 - Bild zurueckrechnen, durch das q-te Perzentil der
    Hirnvoxel teilen statt durch das Maximum, wieder invertieren, auf [1e-6, 1] kappen. Hintergrund bleibt 0."""
    def neu(*a, **kw):
        lade(*a, **kw)
        for r in cv5.FA["recs"].values():
            h = r["img"] != 0
            if h.any():
                roh = 1.0 - r["img"][h]
                p = max(float(np.percentile(roh, q)), 1e-6)
                r["img"][h] = np.clip(1.0 - roh / p, 1e-6, 1.0)
        print(f"Perzentil-Normierung ({q}) je Fall im Hirn angewandt", flush=True)
    return neu


LANDMARKEN_PERZENTILE = (10, 25, 50, 75, 90, 99)


def landmarken_bestimmen(bilder) -> list:
    """Referenz-Landmarken: Median der Fall-Perzentile ueber die gegebenen Bilder (auf der GELIEFERTEN, invertierten Skala)."""
    return list(np.median([np.percentile(b[b != 0], LANDMARKEN_PERZENTILE) for b in bilder], axis=0))


def mit_landmarken_normierung(lade, referenz_datei):
    """Stueckweise lineare Angleichung an feste Referenz-Landmarken (Nyul & Udupa 1999), je Fall im Hirn.

    WARUM (gemessen 23.09.2026, cmb/analysis/harmonisation_check.py): der Unterschied zwischen unseren Kohorten ist
    NICHTLINEAR. Alle affinen Skalen scheitern daran -- durch das Maximum zu teilen laesst die Spanne des
    Laesionskontrasts zwischen den Kohorten bei 0.060, das 99.5-Perzentil bei 0.057, und die z-Normierung macht sie
    mit 0.362 sogar sechsmal SCHLECHTER (wer durch die Hirn-Streuung teilt, gleicht die Verteilung an und verzerrt
    dabei den Laesionskontrast je Kohorte verschieden). Die Landmarken-Angleichung bringt sie auf 0.018 und gleicht
    Niveau und Streuung exakt an -- das einzige Verfahren, das einen nichtlinearen Unterschied beheben kann.

    Wie: je Fall die Perzentile 10/25/50/75/90/99 der Hirnvoxel bestimmen und stueckweise linear auf die hinterlegten
    Referenzwerte abbilden (np.interp; ausserhalb der Landmarken wird konstant fortgesetzt). Hintergrund bleibt exakt 0,
    Hirnvoxel, die sonst 0 wuerden, bekommen 1e-6 -- dieselbe Konvention wie bei den anderen Skalen, damit
    "Kachel ausserhalb" und "Hirnvoxel" weiter stimmen.

    Die Referenzdatei MUSS dieselbe sein wie beim Training; sonst rechnet die Vorhersage auf einer anderen Skala.
    Erzeugen: python code/landmarken_bauen.py --gitter dev/gitter_d --aus dev/landmarken_valdo.json
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
        print(f"Landmarken-Normierung je Fall im Hirn angewandt (Referenz {os.path.basename(referenz_datei)})", flush=True)
    return neu


def mit_z_normierung(lade):
    """z-Normierung des Bildkanals je Fall ueber die Hirnvoxel (Bild != 0); Hintergrund bleibt exakt 0,
    damit 'Kachel ganz ausserhalb' (proba_fall) und 'Hirnvoxel' (stichprobe) weiter stimmen. Hirnvoxel,
    die durch die Normierung exakt 0 wuerden, bekommen 1e-6."""
    def neu(*a, **kw):
        lade(*a, **kw)
        for r in cv5.FA["recs"].values():
            h = r["img"] != 0
            if h.any():
                v = r["img"][h]; v = (v - v.mean()) / max(float(v.std()), 1e-6)
                v[v == 0] = 1e-6
                r["img"][h] = v
        print("z-Normierung je Fall im Hirn angewandt", flush=True)
    return neu


def mit_augmentierung(stichprobe):
    """Nach der Rezept-Stichprobe (mit Spiegelungen) auf der GPU, stapelweise: Drehung um die z-Achse
    +-15 Grad und Massstab 0.9-1.1 (Bild bilinear, Label naechster Nachbar -- eine CMB von wenigen Voxeln
    darf nicht zu Bruchteilen verschmiert werden), dann nur auf dem Bildkanal 0 und nur im Hirn:
    Kontrast x(0.9-1.1) um das Patch-Mittel, Helligkeit +-0.1 Std, Rauschen bis 0.05 Std."""
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
    """NUR Intensitaet, KEINE Geometrie (21.09.2026): nichts wird interpoliert, eine Blutung von wenigen Voxeln bleibt scharf. Je Ausschnitt, jeweils mit
    Wahrscheinlichkeit 0.5, nur Bildkanal 0 und nur im Hirn: Gamma 0.7-1.5 auf der invertierten Skala, glattes multiplikatives Feld (3x3x3-Gitter,
    exp(N(0, 0.15)), trilinear) wie ein Rest-Biasfeld, Kontrast x(0.85-1.15) um das Hirnmittel des Ausschnitts, Helligkeit +-0.1 Std, Rauschen bis 0.05 Std.
    Der FRST-Kanal bleibt unveraendert (er ist auf relativen Kontrast genormt). Ziel: Scanner- und Skalenunterschiede zwischen Kohorten."""
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
    """Dickschicht-Simulation (21.09.2026): mit Wahrscheinlichkeit 0.35 wird ein Ausschnitt so behandelt, als waere er mit k = 2..5 mm Schichtdicke aufgenommen
    und auf 1 mm zurueckgerechnet: Mittel ueber k Schichten, dann lineare Interpolation in z (Bild und FRST). Das Label folgt der Aufnahme: eine Blutung, die eine
    dicke Schicht beruehrt, zaehlt in der ganzen Schicht (Maximum ueber k, naechster Nachbar zurueck). Hintergrund bleibt 0. Ziel: die duennschichtigen Faelle
    (VALDO-Kohorte 2) liefern Trainingsbeispiele fuer das Aussehen der dickschichtigen (Kohorten 1 / 3 und die zweite Kohorte), wo wir am schwaechsten sind."""
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
    """Wie cv5.falte_ordner: nur meta.ok gilt als fertig, Unvollstaendiges wird beiseitegelegt."""
    outd = f"{basis}/fold{k}"
    mp = f"{outd}/meta.json"
    if os.path.exists(mp) and json.load(open(mp)).get("ok"):
        return outd, True
    if os.path.isdir(outd) and os.listdir(outd):
        ab = f"{outd}_abgebrochen-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(outd, ab)
        print(f"unvollstaendige Falte beiseitegelegt: {ab}", flush=True)
    os.makedirs(outd, exist_ok=True)
    return outd, False


def eine_falte(netz, k, outd, **kw):
    """train_falte mit einem OOM-Auffang: Stapel 4 -> 2 (mit InstanceNorm aendert das die
    Normierung nicht, nur die Zahl der Schritte). Wird in meta.json vermerkt."""
    abw = None
    _INIT["fold"] = k
    try:
        m = cv5.train_falte(netz, k, outd, **kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        abw = 2
        print(f"[{netz} f{k}] CUDA-OOM bei Stapel {cv5.REZEPT['bs']} -> Wiederholung mit Stapel 2", flush=True)
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
                    help="t2s = Kontrolle ohne FRST-Kanal (Ordner bekommt den Zusatz -t2s)")
    ap.add_argument("--epochen", type=int, default=None,
                    help="Rezept 2 (19.09.): feste Trainingsdauer statt Fruehstopp -- max_ep = min_ep = N, "
                         "Auswertung der inneren Falte alle N/12 Epochen; Ordnerzusatz -e<N>")
    ap.add_argument("--anteil-laesion", type=float, default=None,
                    help="Anteil der Patches MIT Blutung (Rezept 0.60 = 1.5:1). Die Negativen behalten das "
                         "Verhaeltnis Hirn:beliebig = 25:15. 0.50 = 1:1, 0.333 = 1:2; Ordnerzusatz -l<Prozent>")
    ap.add_argument("--landmarken", default="dev/landmarken_valdo.json",
                    help="Referenzdatei fuer --norm landmarken; MUSS bei Training und Vorhersage dieselbe sein")
    ap.add_argument("--norm", choices=["max", "z", "p995", "landmarken"], default="max",
                    help="z = z-Normierung des T2*-Kanals je Fall im Hirn (wie nnU-Net), Hintergrund bleibt 0; "
                         "max = wie geliefert (1 - Bild/Maximum). Ordnerzusatz -nz")
    ap.add_argument("--aug", action="store_true",
                    help="zusaetzliche Augmentierung (GPU): Drehung in der Schicht +-15 Grad, Massstab 0.9-1.1, "
                         "Helligkeit/Kontrast +-10 %%, Rauschen; Spiegelungen bleiben. Ordnerzusatz -aug")
    ap.add_argument("--aug-intensitaet", action="store_true",
                    help="nur Intensitaets-Augmentierung (Gamma, glattes Feld, Kontrast, Helligkeit, Rauschen), keine Interpolation; Ordnerzusatz -augi")
    ap.add_argument("--aug-schicht", action="store_true",
                    help="Dickschicht-Simulation in z (k = 2..5 mm, Wahrscheinlichkeit 0.35), Label folgt; Ordnerzusatz -augs")
    ap.add_argument("--hp", nargs="*", default=[], metavar="SCHLUESSEL=WERT",
                    help="Turnier 2: Rezeptwerte ueberschreiben (lr, weight_decay, bs, n_patches, anteil_laesion, anteil_hirn); Ordnerzusatz -hp<...>")
    ap.add_argument("--verlust", choices=["bce_dice", "blob", "zentrum", "fnrw", "focal_tversky", "bce_dice_ohne_ds"], default="bce_dice",
                    help="blob = BCE+Dice plus ein Term je Blutung (blob loss, Kofler 2023); zentrum = BCE+Dice plus Zentrumskarte als "
                         "Hilfsausgabe (CenSynCMB 2026), nur mit --netz a12-anisozentrum; fnrw = Ausschnitte nach Groesse und aktueller "
                         "Netzantwort ihrer Blutungen gewichtet (CenSynCMB); Ordnerzusatz -v<Name>")
    ap.add_argument("--gewebe", nargs="?", const="dev/synthseg/{id}_synthseg.nii.gz", default=None, metavar="MUSTER",
                    help="fuenf One-hot-Gewebekanaele aus der SynthSeg-Karte als zusaetzliche Eingabe (Liquor, Rinde, Mark, tiefe Kerne, "
                         "infratentoriell); MUSTER mit {id} oder {sid}, Vorgabe VALDO; Ordnerzusatz -gew")
    ap.add_argument("--letzter-stand", action="store_true",
                    help="Gegenprobe 21.09.: die Testvorhersage kommt vom LETZTEN Zwischenstand (wirklich feste Epochenzahl) statt vom besten "
                         "Zwischenstand der inneren Falte; Schwelle / Mindestgroesse weiter von der inneren Falte. Ordnerzusatz -ls")
    ap.add_argument("--synthese", default=None, metavar="GITTER",
                    help="Gitter mit kuenstlichen Blutungen/Verwechslern (cmb.synthesis.build_synthetic_grid); nur Trainingsfaelle; Ordnerzusatz -syn")
    ap.add_argument("--synthese-anteil", type=float, default=None, metavar="ANTEIL",
                    help="mit --synthese: zweite Fassung -- echte Stichprobe unveraendert, ANTEIL der laesionsfreien Ausschnitte durch kuenstliche Blutungen ersetzt; Ordnerzusatz -syn<Prozent>")
    ap.add_argument("--zusatz", nargs="+", default=None, metavar="NAME",
                    help="stetige Zusatzkanaele <id>_<NAME>.nii.gz aus dem Gitterordner (z. B. t1 t2; code/zusatzkanaele_bauen.py); Ordnerzusatz -z<NAMEN>")
    ap.add_argument("--harte-negative", action="store_true",
                    help="ab Epoche 11 alle 10 Epochen die eigenen Fehlalarme auf den Trainingsfaellen schuerfen und 60 %% der "
                         "blutungsfreien Ausschnitte dorthin legen; Ordnerzusatz -hn")
    ap.add_argument("--lr", type=float, default=None, help="Lernrate statt 3e-4; Ordnerzusatz -lr<Wert>")
    ap.add_argument("--maske", choices=["valdo", "synthseg"], default="valdo",
                    help="synthseg = Bild und FRST ausserhalb des SynthSeg-Hirns (um 2 Voxel geweitet) auf 0 setzen. "
                         "Die gelieferte VALDO-Maske ist in Kohorte 3 zu 40 %% Nicht-Hirn (Augen, Schaedelbasis), "
                         "in Kohorte 2 zu 16 %% (gemessen 19.09.). Ordnerzusatz -ms")
    ap.add_argument("--manifest", nargs="*", default=None,
                    help="19.09. Transfer: eine oder mehrere JSON-Listen [{id, fold, dir, cohort}, ...] statt dev/cases.json + "
                         "dev/split.json. Faelle mehrerer Kohorten werden gemeinsam in 5 Falten trainiert (Falte = Feld fold).")
    ap.add_argument("--name", default=None, help="Ordnerzusatz fuer --manifest/--init-Laeufe, z.B. mix-t2s")
    ap.add_argument("--init", default=None,
                    help="Feintuning: Startgewichte je Falte, Platzhalter {fold}, z.B. .../a03-aniso-e60-gd/fold{fold}/modell.pt")
    ap.add_argument("--basis", default=None, help="Ergebnisordner statt ergebnisse/turnier; private Kohorten bekommen hier einen Ordner ausserhalb dieses Baums")
    ap.add_argument("--gitter", default=None,
                    help="anderes Datengitter statt dev/gitter_c, z.B. dev/gitter_d (ohne Gefaess-Uebermalung); "
                         "Ordnerzusatz -g<Name>")
    ap.add_argument("--patch", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"),
                    help="Patchgroesse in Voxeln (z y x) statt cv5.PATCH (38 62 94); Ordnerzusatz -p<Z>x<Y>x<X>")
    ap.add_argument("--bs", type=int, default=None, help="nur --probe: Stapel (auch fuer die Kacheln)")
    ap.add_argument("--vram-anteil", type=float, default=None,
                    help="nur --probe: Deckel fuer den eigenen VRAM-Anteil (schuetzt den Modellserver)")
    a = ap.parse_args()
    if a.liste:
        for n in NETZE:
            print(n)
        return
    for n in a.netz:
        assert n in NETZE, f"unbekanntes Netz {n!r}; --liste"
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
        assert a.epochen, "--letzter-stand braucht --epochen (fester Plan)"
        cv5.LETZTER_STAND = True
        zusatz += "-ls"
    if a.lr:
        for n in a.netz:
            cv5.KONFS[n]["lr"] = a.lr
        zusatz += f"-lr{a.lr:g}"
    if ("a12-anisozentrum" in a.netz) != (a.verlust == "zentrum"):
        raise SystemExit("--netz a12-anisozentrum und --verlust zentrum gehoeren zusammen (und nur zusammen)")
    if a.verlust != "bce_dice":
        for n in a.netz:
            cv5.KONFS[n]["loss"] = "bce_dice" if a.verlust == "bce_dice_ohne_ds" else a.verlust
        zusatz += "-v" + a.verlust
    for hp in a.hp:                                       # Turnier 2: ein Rezeptwert je Lauf
        k, v = hp.split("=", 1)
        assert k in ("lr", "weight_decay", "bs", "n_patches", "anteil_laesion", "anteil_hirn"), k
        val = int(v) if k in ("bs", "n_patches") else float(v)
        for n in a.netz:
            cv5.KONFS[n][k] = val
        zusatz += f"-hp{k.replace('_', '')}{v}"
    if a.maske == "synthseg":
        cv5.lade_daten = mit_synthseg_maske(cv5.lade_daten)     # VOR der z-Normierung: die rechnet dann nur im Hirn
        zusatz += "-ms"
    if a.manifest:
        if a.maske != "valdo" or a.gitter:
            raise SystemExit("--manifest vertraegt sich (noch) nicht mit --maske/--gitter: der Manifest-Lader ersetzt deren Lader")
        cv5.lade_daten = mit_manifest(a.manifest)
    # Die Normierung wird ZULETZT umgelegt, damit sie den tatsaechlich benutzten Lader umschliesst -- egal ob Gitter
    # oder Manifest. Vorher stand sie davor und wurde vom Manifest-Lader still ueberschrieben; deshalb war
    # "--manifest + --norm" verboten. Genau die Kombination braucht das Mischen mit Angleichung (23.09.2026).
    if a.norm == "z":
        cv5.lade_daten = mit_z_normierung(cv5.lade_daten)
    if a.norm == "p995":
        cv5.lade_daten = mit_perzentil_normierung(cv5.lade_daten)
    if a.norm == "landmarken":
        cv5.lade_daten = mit_landmarken_normierung(cv5.lade_daten, a.landmarken if os.path.isabs(a.landmarken) else f"{ROOT}/{a.landmarken}")
    if a.gewebe:                                          # NACH der Wahl des Laders (Gitter oder Manifest): haengt die Karte an jeden Fall
        if a.harte_negative or a.aug or a.aug_schicht or a.aug_intensitaet or a.probe:
            raise SystemExit("--gewebe vertraegt sich (noch) nicht mit --harte-negative / --aug / --probe: deren Stichprobe kennt die Zusatzkanaele nicht")
        _GEWEBE["an"] = True
        cv5.lade_daten = mit_gewebekanal(cv5.lade_daten, a.gewebe)
        cv5.stichprobe, cv5.proba_fall = stichprobe_gewebe, proba_fall_gewebe
        zusatz += "-gew"
    if a.zusatz:                                          # NACH der Wahl des Laders, wie --gewebe
        if a.harte_negative or a.aug or a.aug_schicht or a.aug_intensitaet or a.probe or a.manifest or a.synthese:
            raise SystemExit("--zusatz vertraegt sich (noch) nicht mit --harte-negative / --aug / --probe / --manifest / --synthese")
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
        cv5.lade_daten(ids=set(folds[0][:2] + folds[1][:2] + folds[2][:3]))   # 7 Faelle reichen
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
                print(f"[{name} f{k}] fertig -- uebersprungen", flush=True); continue
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
                print(f"[{name} f{k}] FEHLER: {eintrag['fehler']}", flush=True)
            # Status frisch lesen: mehrere Netz-Laeufe schreiben nacheinander in dieselbe Datei
            status = json.load(open(status_p)) if os.path.exists(status_p) else {}
            status[f"{name}/fold{k}"] = eintrag
            json.dump(status, open(status_p, "w"), indent=1)
    print(f"TURNIER {' '.join(a.netz)}: {fehler} Falte(n) mit Fehler.", flush=True)
    sys.exit(1 if fehler else 0)


if __name__ == "__main__":
    main()
