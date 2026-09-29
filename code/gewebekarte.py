#!/usr/bin/env python3
"""Tissue map + CMB (Claude 18 Sep 2026, user request: output GM / WM / CSF / CMB instead of just CMB).

Path A -- without new training: the SynthSeg map (dev/synthseg/<id>_synthseg.nii.gz, grid C,
FreeSurfer label values) is combined into three tissues, the CMB prediction of a network
is added on top as a fourth class:

    0 background   1 CSF   2 GM   3 WM   4 CMB

For this, per predicted component the LOCATION (majority of the SynthSeg labels under the
component; if it lies entirely in the background, the nearest label within 3 voxels counts):
    lobar (cortex + cerebral WM), deep (thalamus, basal ganglia, ventral DC),
    infratentorial (brainstem, cerebellum), mesiotemporal (hippocampus, amygdala),
    CSF (ventricles, free CSF), outside.
A boundary that belongs in the report: deep white matter (internal/external capsule, corpus
callosum, periventricular) is not separated by SynthSeg from the rest of the WM -- such CMB
count here as lobar.

LABEL TABLE = FreeSurferColorLUT, checked 18 Sep against the values in the files
(0,2,3,4,5,7,8,10-18,24,26,28,41-44,46,47,49-54,58,60). NOTE: code/we5.py takes as
"ventricle" {4,12,13,22,23} and as CSF 17 -- in these files that is the left lateral
ventricle, left PUTAMEN, left PALLIDUM (22/23 do not occur) and left HIPPOCAMPUS. The we5
measurement "T1 filter costs 0.0090 F1" is therefore invalid; the table here is the
correction, and the location table below (TP/FP per location) is the clean re-measurement
of the question.

Usage:
  python gewebekarte.py --netz ergebnisse/turnier/a01-attunet [--ohne-karten]
  -> <netz>/gewebe/<id>_gewebe_cmb.nii.gz (uint8, grid C) and <netz>/gewebe/orte.json
     (SUMS ONLY: per location number of predictions, of which TP components and FP; reference CMB per location).
"""
import os, sys, json, glob, argparse
import numpy as np
import nibabel as nib
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
N26 = np.ones((3, 3, 3), bool)

CSF = {4, 5, 14, 15, 24, 43, 44}                       # lateral/temporal horn L+R, 3rd, 4th ventricle, free CSF
WM = {2, 41, 7, 46, 16}                                # cerebral WM, cerebellar WM, brainstem
GM = {3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60}
ORT = {}
for _l in (2, 41, 3, 42):                 ORT[_l] = "lobaer"
for _l in (10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60): ORT[_l] = "tief"
for _l in (7, 8, 16, 46, 47):             ORT[_l] = "infratentoriell"
for _l in (17, 18, 53, 54):               ORT[_l] = "mesiotemporal"
for _l in CSF:                            ORT[_l] = "liquor"
ORTE = ("lobaer", "tief", "infratentoriell", "mesiotemporal", "liquor", "ausserhalb")
assert not (CSF & WM or CSF & GM or WM & GM) and set(ORT) == CSF | WM | GM


def gewebe(seg):
    g = np.zeros(seg.shape, np.uint8)
    g[np.isin(seg, list(CSF))] = 1
    g[np.isin(seg, list(GM))] = 2
    g[np.isin(seg, list(WM))] = 3
    return g


def ort_der_komponente(maske, seg):
    """Majority label under the component; background only -> dilate the component by 3 voxels."""
    for it in (0, 3):
        m = ndimage.binary_dilation(maske, structure=N26, iterations=it) if it else maske
        w = seg[m]; w = w[w > 0]
        if len(w):
            return ORT.get(int(np.bincount(w).argmax()), "ausserhalb")
    return "ausserhalb"


def orte_zaehlen(bin_maske, seg, andere=None):
    """-> {location: n} and, if `andere` is given, {location: n touching `andere`}."""
    lab, n = ndimage.label(bin_maske, structure=N26)
    alle = dict.fromkeys(ORTE, 0); treffer = dict.fromkeys(ORTE, 0)
    if n == 0:
        return alle, treffer
    sl = ndimage.find_objects(lab)
    for c in range(1, n + 1):
        s = tuple(slice(max(a.start - 4, 0), a.stop + 4) for a in sl[c - 1])
        m = lab[s] == c
        o = ort_der_komponente(m, seg[s])
        alle[o] += 1
        if andere is not None and (andere[s] & m).any():
            treffer[o] += 1
    return alle, treffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netz", required=True, help="folder with fold<k>/<id>_pred.nii.gz")
    ap.add_argument("--ohne-karten", action="store_true", help="only orte.json, do not write NIfTI")
    a = ap.parse_args()
    netz = a.netz if os.path.isabs(a.netz) else f"{ROOT}/{a.netz}"
    aus = f"{netz}/gewebe"; os.makedirs(aus, exist_ok=True)
    pred_n = dict.fromkeys(ORTE, 0); pred_tp = dict.fromkeys(ORTE, 0)
    ref_n = dict.fromkeys(ORTE, 0); ref_gef = dict.fromkeys(ORTE, 0)
    n_faelle, ohne_seg = 0, 0
    for f in sorted(glob.glob(f"{netz}/fold*/*_pred.nii.gz")):
        sid = os.path.basename(f)[:-len("_pred.nii.gz")]
        sp = f"{ROOT}/dev/synthseg/{sid}_synthseg.nii.gz"
        if not os.path.exists(sp):
            ohne_seg += 1; continue
        pn = nib.load(f)
        pred = np.asarray(pn.dataobj) > 0
        seg = np.asarray(nib.load(sp).dataobj).astype(np.int32)
        ref = np.asarray(nib.load(f"{ROOT}/dev/gitter_c/{sid}/{sid}_label.nii.gz").dataobj) > 0
        assert pred.shape == seg.shape == ref.shape, sid
        n_faelle += 1
        pa, pt = orte_zaehlen(pred, seg, ref)
        ra, rt = orte_zaehlen(ref, seg, pred)
        for o in ORTE:
            pred_n[o] += pa[o]; pred_tp[o] += pt[o]; ref_n[o] += ra[o]; ref_gef[o] += rt[o]
        if not a.ohne_karten:
            g = gewebe(seg); g[pred] = 4
            nib.save(nib.Nifti1Image(g, pn.affine), f"{aus}/{sid}_gewebe_cmb.nii.gz")
    orte = {o: dict(vorhersagen=pred_n[o], davon_auf_referenz=pred_tp[o], fp=pred_n[o] - pred_tp[o],
                    referenz_cmb=ref_n[o], davon_gefunden=ref_gef[o]) for o in ORTE}
    erg = dict(netz=os.path.relpath(netz, ROOT), n_faelle=n_faelle, faelle_ohne_synthseg=ohne_seg,
               klassen={"0": "background", "1": "CSF", "2": "GM", "3": "WM", "4": "CMB"},
               grenze="deep white matter counts as lobar (SynthSeg does not separate it)", orte=orte)
    json.dump(erg, open(f"{aus}/orte.json", "w"), indent=1, ensure_ascii=False)
    print(f"{erg['netz']}: {n_faelle} cases, {ohne_seg} without SynthSeg")
    print(f"{'Location':16s} {'Pred.':>8s} {'on ref':>8s} {'FP':>5s} | {'Ref CMB':>7s} {'found':>8s}")
    for o in ORTE:
        e = orte[o]
        print(f"{o:16s} {e['vorhersagen']:8d} {e['davon_auf_referenz']:8d} {e['fp']:5d} | "
              f"{e['referenz_cmb']:7d} {e['davon_gefunden']:8d}")


if __name__ == "__main__":
    main()
