"""wk1b: MedNet-CMB Zero-Shot Inference auf den In-house-Kohorten (GPU, ueber eine GPU-Warteschlange).

Modell: MedNet-CMB (Apache-2.0), nnU-Net v2 3d_fullres, 5 Falten, Dataset012_ASPREE_VALDO_CSIRO.
Vorverarbeitung = exakt wie das README/Repo (mednet_cmb/preprocess.py), ABER:
  - Hirnmaske = unsere HD-BET-Masken (<cohort>/derivatives/brainmask), nicht SynthStrip
    (nipreps-synthstrip ist in keinem Env installiert -> Deviation, siehe Manifest).
  - N4 (SimpleITK, auf shrunkem Bild, Bias-Field auf volle Res aufgeteilt), 2-Voxel-Dilatation,
    Perzentil-Clip [0.5; 99.5] im Hirn, Aussen 0, Res [0,1]. Geometrie bleibt erhalten.
Wahrscheinlichkeitskarte: predict_from_files(save_probabilities=True) -> .npz, CMB-Klasse = Channel 1.
Ausgabe: <out>/<cohort>/<id>_prob.nii.gz (float32, auf dem ORIGINAL-Bildgitter, CMB-Klassen-Wahrscheinlichkeit).
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
import SimpleITK as sitk

REPO = _EXTMODELS + "/mednet_cmb/repo"
W = _EXTMODELS + "/mednet_cmb/MedNet-CMB_weights"
MODEL_DIR = f"{W}/nnUNet_results/Dataset012_ASPREE_VALDO_CSIRO_stratified_1x1x1_relabeled_v2/nnUNetTrainer__nnUNetPlans__3d_fullres"
FAELLE = _CASES
OUT = _P + "/extern"
sys.path.insert(0, REPO)

N4_ITERS = (50, 50, 30, 20); N4_CTRL = (4, 4, 4)


def preprocess_case(bild, hirn, dst):
    """MedNet-Vorverarbeitung mit UNSERER Hirnmaske (statt SynthStrip). Gibt float32 [0,1] zurueck,
    Geometrie identisch zum Eingangsbild."""
    img = sitk.ReadImage(bild, sitk.sitkFloat32)
    if hirn:
        mask = sitk.Cast(sitk.ReadImage(hirn) > 0.5, sitk.sitkUInt8)
    else:
        # VALDO: Bild ist bereits hirnextrahiert (desc-masked) -> Maske aus dem Bild
        mask = sitk.Cast(sitk.Cast(img > 0, sitk.sitkUInt8), sitk.sitkUInt8)
    if tuple(mask.GetSize()) != tuple(img.GetSize()):
        mask = sitk.Resample(mask, img, sitk.Resampler(), sitk.sitkNearestNeighbor)
    mask = sitk.BinaryDilate(mask, [2, 2, 2])
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    mask_u8 = sitk.Cast(mask > 0, sitk.sitkUInt8)
    img_s = sitk.Shrink(img_f, [2, 2, 2]); mask_s = sitk.Shrink(mask_u8, [2, 2, 2])
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(list(N4_ITERS)); n4.SetNumberOfControlPoints(list(N4_CTRL))
    n4.Execute(img_s, mask_s)
    log_bias = n4.GetLogBiasFieldAsImage(img_f)
    corrected = sitk.Divide(img_f, sitk.Exp(log_bias))
    masked = sitk.Mask(corrected, mask)
    arr = sitk.GetArrayFromImage(masked).astype(np.float32)
    m = sitk.GetArrayFromImage(mask) > 0
    vals = arr[m] if np.any(m) else arr
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if hi <= lo: hi = lo + 1e-6
    arr[m] = (np.clip(arr[m], lo, hi) - lo) / (hi - lo); arr[~m] = 0.0
    out = sitk.GetImageFromArray(arr); out.CopyInformation(img)
    assert np.allclose(img.GetSpacing(), out.GetSpacing()) and img.GetSize() == out.GetSize()
    sitk.WriteImage(out, dst, True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["swi", "t2star", "valdo"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = alle; >0 = nur die ersten N (Probelauf)")
    args = ap.parse_args()
    t0 = time.time()
    if args.cohort == "valdo":
        man = json.load(open(_R + "/dev/manifest_valdo_gitter_d.json"))
        cases = [dict(id=e["id"], bild=f"{_VALDO}/{e['id']}/{e['id']}_space-T2S_desc-masked_T2S.nii.gz", hirn=None) for e in man]
    else:
        fa = json.load(open(FAELLE))[args.cohort]["faelle"]
        cases = [dict(id=c["id"], bild=c["bild"], hirn=c["hirn"]) for c in fa]
    if args.limit:
        cases = cases[:args.limit]
    in_dir = f"{OUT}/mednet-{args.cohort}-in"; os.makedirs(in_dir, exist_ok=True)
    out_dir = f"{OUT}/mednet-{args.cohort}"; os.makedirs(out_dir, exist_ok=True)
    for c in cases:
        src = f"{in_dir}/{c['id']}_0000.nii.gz"
        if not os.path.exists(src):
            preprocess_case(c["bild"], c["hirn"], src)
        print(f"preproc {c['id']}", flush=True)

    # 04.11.2026: nnU-Net v2.8 has no PredictFromRawData (ImportError in 4 queue runs); the predictor class is
    # nnUNetPredictor, and the MedNet-CMB release ships folds 0-4 (checkpoint_final.pth each), no "all" fold.
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=False)
    pred.initialize_from_trained_model_folder(MODEL_DIR, use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_final.pth")
    prob_dir = f"{out_dir}/prob"; os.makedirs(prob_dir, exist_ok=True)
    pred.predict_from_files(in_dir, prob_dir, save_probabilities=True,
                            overwrite=True,
                            num_processes_preprocessing=3, num_processes_segmentation_export=3)
    # npz -> CMB-Klasse (Channel 1) als NIfTI auf dem Originalgitter
    for c in cases:
        npz = f"{prob_dir}/{c['id']}_0000.probabilities.npy"
        if not os.path.exists(npz):
            cand = sorted(f for f in os.listdir(prob_dir) if c["id"] in f and f.endswith(".npy"))
            if not cand:
                print("FEHLT", c["id"], os.listdir(prob_dir)[:5], flush=True); continue
            npz = f"{prob_dir}/{cand[0]}"
        probs = np.load(npz)["probabilities"]
        cmb = probs[1] if probs.shape[0] > 1 else probs[0]
        ref = nib.load(c["bild"])
        if cmb.shape != np.asarray(ref.dataobj).shape:
            from nibabel.processing import resample_from_to
            r = resample_from_to(nib.Nifti1Image(cmb.astype(np.float32), ref.affine),
                                 (np.asarray(ref.dataobj).shape, ref.affine), order=1)
            cmb = np.asarray(r.dataobj)
        nib.save(nib.Nifti1Image(cmb.astype(np.float32), ref.affine), f"{out_dir}/{c['id']}_prob.nii.gz")
        print(f"prob {c['id']}: max={cmb.max():.4f}, >0.05: {(cmb>0.05).sum()}", flush=True)
    json.dump(dict(cohort=args.cohort, n=len(cases), dauer_s=round(time.time() - t0),
                   model_dir=MODEL_DIR, mask="HD-BET (ours), nicht SynthStrip (Deviation)"),
              open(f"{out_dir}/lauf.json", "w"), indent=1)
    print(json.dumps(dict(cohort=args.cohort, n=len(cases), dauer_s=round(time.time() - t0))))


if __name__ == "__main__":
    main()
