"""wk1b: MicrobleedNet ORIGINAL (gelieferte Gewichte, KEIN Feintuning) Zero-Shot auf den In-house-Kohorten
(GPU, ueber eine GPU-Warteschlange).

Original-Pipeline (microbleednet-Paket, github.com/v-sundaresan/microbleed-detection):
  - preprocess_subject: eigene Vorverarbeitung (vessels inpaint, FRST, Inversion), Geometrie erhalten.
  - CDet (48-Voxel-Non-Overlapping-Patches, Softmax, Kandidaten = prob > 0.1 -- 0.1 ist die im Paket
    dokumentierte Schwelle der gelieferten Gewichte, s. cdet_evaluate_function.py, Diagnose 07.09.):
    Wahrscheinlichkeitskarte statt binarem Patch-Stack (put_patches_into_volume auf prob, nicht > 0.1).
  - CDisc (Patches um CDet-Komponenten, Softmax [0,1] = P(CMB) je Komponente) ->
    Wahrscheinlichkeit je Komponente, sonst 0; auf das volle (un-gedempfte) preproc-Volumen zurueck.
  Kein shape_based_filtering / kein Argmax: das ist Post-Processing, wir liefern die
  Wahrscheinlichkeitskarte und post-processen zentral nach den Arenaregeln (v2_postprocess-Logik).
Ausgabe: <out>/<cohort>/<id>_prob.nii.gz (float32, P(CMB) je Voxel, preproc-Gitter = Bildgitter).
"""
from __future__ import annotations
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import argparse, json, os, sys, time
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch import nn

from microbleednet.scripts import data_preparation as dp
from microbleednet.scripts import utils as mutils
import microbleednet.scripts.model_architectures as models

FAELLE = _CASES
OUT = _P + "/extern"
WEIGHTS = _EXTMODELS + "/microbleednet_checkpoints"
CDISC_PATCH, CDET_PATCH, CDET_T = 24, 48, 0.1


def cases_of(cohort):
    if cohort == "valdo":
        man = json.load(open(_R + "/dev/manifest_valdo_gitter_d.json"))
        return [dict(id=e["id"], bild=f"{_VALDO}/{e['id']}/{e['id']}_space-T2S_desc-masked_T2S.nii.gz", hirn=None) for e in man]
    fa = json.load(open(FAELLE))[cohort]["faelle"]
    return [dict(id=c["id"], bild=c["bild"], hirn=c["hirn"]) for c in fa]


def prep(c, pre_dir):
    img_dir, frst_dir = f"{pre_dir}/images", f"{pre_dir}/frsts"
    os.makedirs(img_dir, exist_ok=True); os.makedirs(frst_dir, exist_ok=True)
    if os.path.exists(f"{img_dir}/{c['id']}_preproc.nii.gz"):
        return
    ni = nib.load(c["bild"]); img = np.asarray(ni.dataobj, dtype=np.float32)
    if img.ndim == 4: img = img[..., 0]
    if c.get("hirn"):
        m = np.asarray(nib.load(c["hirn"]).dataobj) > 0
        m = m[..., 0] if m.ndim == 4 else m
        img = np.where(m, img, 0).astype(np.float32)
    masked = f"{pre_dir}/{c['id']}_masked.nii.gz"
    nib.save(nib.Nifti1Image(img, ni.affine), masked)
    subj = dict(input_path=masked, label_path=None, basename=c["id"],
                image_directory=img_dir, label_directory=f"{pre_dir}/labels", frst_directory=frst_dir)
    image, label, frst = dp.preprocess_subject(subj)
    ref = nib.load(masked); aff, hdr = ref.affine, ref.header
    nib.save(nib.Nifti1Image(np.asarray(image, dtype=np.float32), aff, hdr), f"{img_dir}/{c['id']}_preproc.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(frst, dtype=np.float32), aff, hdr), f"{frst_dir}/{c['id']}_frst.nii.gz")


@torch.no_grad()
def infer(c, pre_dir, out_dir, cdet_model, cdisc_model, device):
    dst = f"{out_dir}/{c['id']}_prob.nii.gz"
    if os.path.exists(dst):
        return
    image, label, frst, coords = dp.load_subject(dict(input_path=f"{pre_dir}/images/{c['id']}_preproc.nii.gz",
                                                      frst_path=f"{pre_dir}/frsts/{c['id']}_frst.nii.gz", basename=c["id"]))
    # 05.11.2026: load_subject liefert das CROPPTE Bild (tight_crop); full_shape muss die
    # ROHE (vollschaellige) Form sein, wie es evaluate_function.py macht (raw_image_shape aus
    # nib.load(...).get_fdata().shape) -- sonst passt replace_into_volume_shape nicht.
    full_shape = np.asarray(nib.load(f"{pre_dir}/images/{c['id']}_preproc.nii.gz").dataobj).shape
    brain_mask = (image > 0).astype(int)
    # CDet: 48-Voxel-Patches, P(CMB-Komponente) pro Patch -> zurueck ins (cropped) Volumen
    data_patches, _, _, _, _ = dp.get_nonoverlapping_patches(image, label, brain_mask, frst, CDET_PATCH)
    data_patches = np.asarray(data_patches)          # (N, 2, 48, 48, 48): Kanal 0 = Bild, Kanal 1 = FRST
    data_patches[data_patches < 0] = 0
    probs = []
    for patch in data_patches:
        # 05.11.2026: die Patches sind BEREITS (2, 48, 48, 48) (Bild+FRST);
        # expand_dims(patch, 0) machte (1, 2, 48, 48, 48) -> conv3d sah 6D -> Fehler bzw.
        # (frueher) Nullkarten. CDetNet erwartet (2, 1, 48, 48, 48): p = expand_dims(patch, 1)?
        # Nein -- torch.from_numpy(patch) hat schon (2,48,48,48); unsqueeze(0) -> Batch.
        p = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).to(device, dtype=torch.float)
        probs.append(nn.Softmax(dim=1)(cdet_model(p))[0, 1].cpu().numpy())
    # 05.11.2026 (Runde 6, BEFUND+FIX): put_patches_into_volume(p, volume_template, s)
    # macht np.zeros_like(volume_template) -- mit dem INT-label-Volumen als Template wurden
    # ALLE Float-Probabiltaeten auf 0 gerundet -> die 136 Karten von [443]/[444] waren Null,
    # obwohl CDet 9 Komponenten > 0.1 fand (ein Fall, probmax 0.172). Template: leeres
    # float32-Volumen in Label-Groesse.
    cdet_vol = dp.put_patches_into_volume(np.stack(probs).astype(np.float32),
                                          np.zeros_like(label, dtype=np.float32), CDET_PATCH)
    cdet_comp = cdet_vol > CDET_T
    # CDisc: P(CMB) je CDet-Komponente (nur zur DOKUMENTATION/PRUEFUNG)
    data_patches, _ = dp.get_patches_centered_on_cmb(image, cdet_comp, frst, CDISC_PATCH)
    comp_probs = []
    for patch in data_patches:
        p = np.expand_dims(patch, 0); p[p < 0] = 0
        p = torch.from_numpy(p).to(device, dtype=torch.float)
        comp_probs.append(float(nn.Softmax(dim=1)(cdisc_model(p))[0, 1].cpu().numpy()))
    # 05.11.2026 (Runde 6, BEFUND -> DEVIATION): die VORTRAINIERTE CDisc (gelieferte
    # Gewichte, ohne Feintuning) klassifiziert auf dieser Kohorte ALLE CDet-Kandidaten mit
    # P(CMB) = 0.0 (Logits [~12.8, ~-12.7], exakt gleich fuer alle Patches/Faelle, CPU und
    # GPU; three SWI and two T2* cases; identisch zu den 07.09-Endvorhersagen in
    # an earlier test run, predicted_final = 0). Die Originalkette CDet->CDisc filtert
    # damit systematisch alles -> n_pred = 0 (ein Ausfall ist kein Ergebnis). Die
    # Wahrscheinlichkeitskarte ist daher die CDET-Karte (P(Komponente) je Voxel, die Stufe,
    # die auf dieser Kohorte misst; 07.09-Diagnose: probmax 0.11-0.22, Schwelle 0.1).
    # DEVIATION "cdisc_alle_negativ" im Manifest; Comp-Statistik unten in lauf.json.
    prob_vol = cdet_vol
    full = dp.replace_into_volume_shape(full_shape, prob_vol.astype(np.float32), coords)
    ref = nib.load(f"{pre_dir}/images/{c['id']}_preproc.nii.gz")
    nib.save(nib.Nifti1Image(full.astype(np.float32), ref.affine), dst)
    print(f"prob {c['id']}: comps={len(comp_probs)}, max={full.max():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["swi", "t2star", "valdo"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    cases = cases_of(args.cohort)
    if args.limit: cases = cases[:args.limit]
    pre_dir, out_dir = f"{OUT}/mbnet-{args.cohort}-pre", f"{OUT}/mbnet-{args.cohort}"
    os.makedirs(out_dir, exist_ok=True)
    for c in cases:
        prep(c, pre_dir)
        print(f"preproc {c['id']}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdet_model = models.CDetNet(n_channels=2, n_classes=2, init_channels=64).to(device)
    mutils.load_model(f"{WEIGHTS}/Microbleednet_cdet_model.pth", cdet_model, mode="weights")
    cdisc_model = models.CDiscStudentNet(n_channels=2, n_classes=2, init_channels=64).to(device)
    # 05.11.2026: der Ordner enthaelt "Microbleednet_cdisc_student_model.pth" (ohne "_weights"),
    # nicht "..._model_weights.pth" -- deshalb sind alle 6 Warteschlangen-Laeufe von 04.11. mit
    # FileNotFoundError gestorben (rueckgabe 1), bevor ein einziger Fall gelaufen war.
    mutils.load_model(f"{WEIGHTS}/Microbleednet_cdisc_student_model.pth", cdisc_model, mode="weights")
    stats = []
    for c in cases:
        infer(c, pre_dir, out_dir, cdet_model, cdisc_model, device)
        a = np.asarray(nib.load(f"{out_dir}/{c['id']}_prob.nii.gz").dataobj)
        stats.append(dict(id=c["id"], probmax=float(a.max())))
    json.dump(dict(cohort=args.cohort, n=len(cases), dauer_s=round(time.time() - t0),
                   weights=WEIGHTS, mode="zero-shot original, no fine-tuning",
                   cdet_threshold_documented=CDET_T,
                   kartentyp="CDet-Karte (P(Komponente) je Voxel); CDisc-Original vortrainiert liefert auf allen Kandidaten P(CMB)=0 (DEVIATION cdisc_alle_negativ)",
                   per_case=stats),
              open(f"{out_dir}/lauf.json", "w"), indent=1)
    n_nz = sum(1 for s in stats if s["probmax"] > 0)
    print(json.dumps(dict(cohort=args.cohort, n=len(cases), dauer_s=round(time.time() - t0),
                          kartennichtnull=n_nz)))


if __name__ == "__main__":
    main()
