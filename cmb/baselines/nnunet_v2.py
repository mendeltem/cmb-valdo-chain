"""Arm 10 (2026-09-29): the real nnU-Net v2 framework as a standardised baseline ("nnU-Net Revisited", Isensee 2024).

Input = the masked VALDO T2* at native resolution and the CMB label, nothing else (no FRST, none of our preprocessing);
nnU-Net plans resampling, normalisation, patch size and augmentation itself. Our five folds are handed over as
splits_final.json so every case is predicted out-of-fold exactly once, like every other run of this project.
Trainer nnUNetTrainer_250epochs (the default is 1000; 250 for budget reasons -- stated openly). Scoring: native lesion metric
(26-connectivity, touch rule) and, resampled onto our grid as <root>/als-lauf/fold<k>/<id>_pred_proba.nii.gz, our fixed cell
+ CSF rule via cmb.analysis.operating_point.

    python -m cmb.baselines.nnunet_v2 prepare          # dataset + plan_and_preprocess + splits (CPU)
    python -m cmb.baselines.nnunet_v2 train --fold k   # GPU (queue with grossauftrag --vram 20); trains and predicts the fold
    python -m cmb.baselines.nnunet_v2 score
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, subprocess, time
import nibabel as nib, numpy as np

ROOT = "/home/uchralt/data/work/valdo-t2s"; EXTERN = "/home/uchralt/data/extern/valdo2021/Task2"
NN = "/home/uchralt/data/work/nnunet"; BIN = "/home/uchralt/miniconda3/envs/dl/bin"
DS_ID, DS = 501, "Dataset501_VALDOCMB"; CONFIG, TRAINER = "3d_fullres", "nnUNetTrainer_250epochs"
ENV = dict(os.environ, nnUNet_raw=f"{NN}/raw", nnUNet_preprocessed=f"{NN}/preprocessed", nnUNet_results=f"{NN}/results", nnUNet_n_proc_DA="6")


def cases():
    return json.load(open(f"{ROOT}/dev/manifest_valdo_gitter_d.json"))


def prepare(a):
    raw = f"{NN}/raw/{DS}"
    for d in ("imagesTr", "labelsTr"):
        os.makedirs(f"{raw}/{d}", exist_ok=True)
    for d in ("preprocessed", "results"):
        os.makedirs(f"{NN}/{d}", exist_ok=True)
    for c in cases():
        cid = c["id"]
        img = nib.load(f"{EXTERN}/{cid}/{cid}_space-T2S_desc-masked_T2S.nii.gz")
        a_ = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32), nan=0.0)
        nib.save(nib.Nifti1Image(a_, img.affine), f"{raw}/imagesTr/{cid}_0000.nii.gz")
        lab = nib.load(f"{EXTERN}/{cid}/{cid}_space-T2S_CMB.nii.gz")
        nib.save(nib.Nifti1Image((np.asarray(lab.dataobj) > 0).astype(np.uint8), lab.affine), f"{raw}/labelsTr/{cid}.nii.gz")
    json.dump({"channel_names": {"0": "T2star"}, "labels": {"background": 0, "cmb": 1}, "numTraining": len(cases()), "file_ending": ".nii.gz",
               "name": DS, "description": "VALDO Task 2, 57 CV cases, masked T2S native, CMB labels; folds from valdo-t2s dev/cases.json"},
              open(f"{raw}/dataset.json", "w"), indent=1)
    r = subprocess.run([f"{BIN}/nnUNetv2_plan_and_preprocess", "-d", str(DS_ID), "--verify_dataset_integrity", "-c", CONFIG, "-np", "6"], env=ENV)
    if r.returncode != 0:
        raise SystemExit("plan_and_preprocess failed")
    splits = [dict(train=[c["id"] for c in cases() if c["fold"] != k], val=[c["id"] for c in cases() if c["fold"] == k]) for k in range(5)]
    json.dump(splits, open(f"{NN}/preprocessed/{DS}/splits_final.json", "w"), indent=1)
    print("splits:", [len(s["val"]) for s in splits], "->", f"{NN}/preprocessed/{DS}/splits_final.json")


def train(a):
    t0 = time.time()
    r = subprocess.run([f"{BIN}/nnUNetv2_train", str(DS_ID), CONFIG, str(a.fold), "-tr", TRAINER, "--npz"], env=ENV)
    if r.returncode != 0:
        raise SystemExit(f"train fold {a.fold} rc={r.returncode}")
    t1 = time.time()
    val_ids = [c["id"] for c in cases() if c["fold"] == a.fold]
    src = f"{NN}/raw/{DS}/imagesTr"; tmp = f"{NN}/predict_in/fold{a.fold}"; out = f"{NN}/predict_out/fold{a.fold}"
    os.makedirs(tmp, exist_ok=True); os.makedirs(out, exist_ok=True)
    for cid in val_ids:
        dst = f"{tmp}/{cid}_0000.nii.gz"
        if not os.path.exists(dst):
            os.symlink(f"{src}/{cid}_0000.nii.gz", dst)
    model = f"{NN}/results/{DS}/{TRAINER}__nnUNetPlans__{CONFIG}"
    r = subprocess.run([f"{BIN}/nnUNetv2_predict", "-i", tmp, "-o", out, "-d", str(DS_ID), "-c", CONFIG, "-tr", TRAINER, "-f", str(a.fold), "--save_probabilities"], env=ENV)
    if r.returncode != 0:
        raise SystemExit(f"predict fold {a.fold} rc={r.returncode}")
    json.dump(dict(fold=a.fold, train_s=round(t1 - t0), predict_s=round(time.time() - t1), trainer=TRAINER, config=CONFIG, n_val=len(val_ids)), open(f"{out}/lauf.json", "w"), indent=1)
    print(open(f"{out}/lauf.json").read())


def score(a):
    from scipy import ndimage
    from nibabel.processing import resample_from_to
    N26 = np.ones((3, 3, 3), bool); tp = fp = fn = 0; n = 0; run = f"{NN}/als-lauf"
    for c in cases():
        cid = c["id"]; p = f"{NN}/predict_out/fold{c['fold']}/{cid}.nii.gz"
        if not os.path.exists(p):
            continue
        n += 1
        pi = nib.load(p); pred = np.asarray(pi.dataobj) > 0
        lab = np.asarray(nib.load(f"{NN}/raw/{DS}/labelsTr/{cid}.nii.gz").dataobj) > 0
        rl, nr = ndimage.label(lab, structure=N26); pl, npd = ndimage.label(pred, structure=N26)
        both = lab & pred; hr = np.zeros(nr + 1, bool); hp = np.zeros(npd + 1, bool)
        hr[np.unique(rl[both])] = True; hp[np.unique(pl[both])] = True; hr[0] = hp[0] = False
        tp += hr.sum(); fn += nr - hr.sum(); fp += npd - hp.sum()
        gi = nib.load(f"{c['dir']}/{cid}_image.nii.gz")
        rs = resample_from_to(nib.Nifti1Image(pred.astype(np.uint8), pi.affine), (gi.shape, gi.affine), order=0)
        d = f"{run}/fold{c['fold']}"; os.makedirs(d, exist_ok=True)
        nib.save(nib.Nifti1Image((np.asarray(rs.dataobj) > 0).astype(np.uint8) * 255, gi.affine), f"{d}/{cid}_pred_proba.nii.gz")
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    res = dict(cases_predicted=n, tp=int(tp), fp=int(fp), fn=int(fn), f1=round(f1, 4), precision=round(tp / max(tp + fp, 1), 4), sensitivity=round(tp / max(tp + fn, 1), 4), fp_per_case=round(fp / max(n, 1), 2), grid_run=run)
    json.dump(res, open(f"{NN}/zusammenfassung.json", "w"), indent=1); print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "train", "score"]); ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args(); {"prepare": prepare, "train": train, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
