"""Arm 6 (user request 2026-09-29): the ORIGINAL MicrobleedNet (Sundaresan et al. 2023, pretrained weights) fine-tuned on our
folds, as the fair baseline against our chain -- same cases, same 5 folds, same lesion metric.

Everything goes through the original package (``microbleednet`` CLI / ``data_preparation.preprocess_subject``): its own
preprocessing (inversion, FRST; ``-f True`` = the masked raw image is used as is), its own two-stage fine-tuning (CDet + CDisc,
all layers unfrozen -- the variant that won the earlier arena, 0.458 vs 0.420 on SWI) and its own evaluation.
Layout (mirrors mb-arena/falten2): <root>/mbnet/<cohort>/preproc/{images,labels,frsts}, <root>/mbnet/<cohort>/falten/fold<k>/{train,test}/{images,labels,frsts},
<root>/mbnet/<cohort>/ergebnisse/fold<k>/{modell,auswertung,lauf.json}. VALDO under valdo-t2s (public), catalina under mb-arena (private).

    python -m cmb.baselines.microbleednet_finetune prepare  --cohort valdo|t2star|swi [--processes 6]
    python -m cmb.baselines.microbleednet_finetune finetune --cohort valdo --fold 0          (GPU, ~75 min; queue with grossauftrag --vram 20)
    python -m cmb.baselines.microbleednet_finetune score    --cohort valdo                    (native lesion metric + our grid metric)
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, sys, time
from multiprocessing import Pool
import nibabel as nib, numpy as np

ROOT = "/home/uchralt/data/work/valdo-t2s"; MB = "/home/uchralt/data/work/mb-arena"; EXTERN = "/home/uchralt/data/extern/valdo2021/Task2"
PYBIN = "/home/uchralt/miniconda3/envs/dl/bin"; WEIGHTS = "/home/uchralt/data/models/vortrainiert/microbleednet/microbleednet"


def root_of(cohort):
    return f"{ROOT}/mbnet/valdo" if cohort == "valdo" else f"{MB}/mbnet/{cohort}"


def cases_of(cohort):
    """id, fold, raw (masked) image, label, and our grid dir / id for the grid metric."""
    if cohort == "valdo":
        man = {e["id"]: e for e in json.load(open(f"{ROOT}/dev/manifest_valdo_gitter_d.json"))}
        return [dict(id=i, fold=e["fold"], image=f"{EXTERN}/{i}/{i}_space-T2S_desc-masked_T2S.nii.gz", label=f"{EXTERN}/{i}/{i}_space-T2S_CMB.nii.gz",
                     grid_id=i, grid_dir=e["dir"]) for i, e in man.items()]
    man = {e["id"].split("-", 1)[1]: e for e in json.load(open(f"{MB}/gitter/manifest_{cohort}.json"))}
    out = []
    for c in json.load(open(f"{MB}/faelle.json"))[cohort]["faelle"]:
        e = man[c["id"]]
        out.append(dict(id=c["id"], fold=int(c["fold"]), image=c["bild"], mask=c["hirn"], label=c["maske"], grid_id=e["id"], grid_dir=e["dir"]))
    return out


def _prep(job):
    c, root = job
    from microbleednet.scripts import data_preparation as dp
    img_dir, lab_dir, frst_dir = (f"{root}/preproc/{d}" for d in ("images", "labels", "frsts"))
    if os.path.exists(f"{img_dir}/{c['id']}_preproc.nii.gz") and os.path.exists(f"{frst_dir}/{c['id']}_frst.nii.gz"):
        return c["id"], "vorhanden"
    os.makedirs(f"{root}/bilder", exist_ok=True)
    ni = nib.load(c["image"]); img = np.asarray(ni.dataobj, dtype=np.float32)
    if img.ndim == 4:
        img = img[..., 0]
    if c.get("mask"):
        m = np.asarray(nib.load(c["mask"]).dataobj) > 0
        m = m[..., 0] if m.ndim == 4 else m
        img = np.where(m, img, 0).astype(np.float32)
    masked = f"{root}/bilder/{c['id']}_img.nii.gz"; nib.save(nib.Nifti1Image(img, ni.affine), masked)
    lab = np.asarray(nib.load(c["label"]).dataobj); lab = lab[..., 0] if lab.ndim == 4 else lab
    os.makedirs(lab_dir, exist_ok=True); labfile = f"{root}/bilder/{c['id']}_label.nii.gz"
    nib.save(nib.Nifti1Image((lab > 0).astype(np.uint8), ni.affine), labfile)
    image, label, frst = dp.preprocess_subject(dict(input_path=masked, label_path=labfile, basename=c["id"], image_directory=img_dir, label_directory=lab_dir, frst_directory=frst_dir))
    ref = nib.load(masked); aff, hdr = ref.affine, ref.header          # the package returns arrays; save them like mb-arena/preprocess_parallel.py
    os.makedirs(img_dir, exist_ok=True); os.makedirs(frst_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(image, dtype=np.float32), aff, hdr), f"{img_dir}/{c['id']}_preproc.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(label, dtype=np.uint8), aff, hdr), f"{lab_dir}/{c['id']}_label.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(frst, dtype=np.float32), aff, hdr), f"{frst_dir}/{c['id']}_frst.nii.gz")
    return c["id"], "ok"


def prepare(a):
    cases = cases_of(a.cohort); root = root_of(a.cohort)
    for d in ("images", "labels", "frsts"):
        os.makedirs(f"{root}/preproc/{d}", exist_ok=True)
    with Pool(a.processes) as pool:
        for cid, st in pool.imap_unordered(_prep, [(c, root) for c in cases]):
            print(cid, st, flush=True)
    names = {"images": "{id}_preproc.nii.gz", "labels": "{id}_label.nii.gz", "frsts": "{id}_frst.nii.gz"}
    link_names = {"images": "{id}_preproc.nii.gz", "labels": "{id}_mask.nii.gz", "frsts": "{id}_frst.nii.gz"}   # fine_tune wants <id>_mask.nii.gz
    for k in range(5):
        for part, sel in (("train", lambda c: c["fold"] != k), ("test", lambda c: c["fold"] == k)):
            for sub, pat in names.items():
                d = f"{root}/falten/fold{k}/{part}/{sub}"; os.makedirs(d, exist_ok=True)
                for c in cases:
                    if sel(c):
                        src = f"{root}/preproc/{sub}/{pat.format(id=c['id'])}"; dst = f"{d}/{link_names[sub].format(id=c['id'])}"
                        assert os.path.exists(src), src
                        if os.path.lexists(dst):
                            os.remove(dst)
                        os.symlink(src, dst)
        print(f"fold{k}: train {sum(c['fold'] != k for c in cases)}, test {sum(c['fold'] == k for c in cases)}")
    json.dump(cases, open(f"{root}/faelle.json", "w"), indent=1)


def finetune(a):
    root = root_of(a.cohort); F = f"{root}/falten/fold{a.fold}"; Z = f"{root}/ergebnisse/fold{a.fold}"
    os.makedirs(f"{Z}/modell", exist_ok=True); os.makedirs(f"{Z}/auswertung", exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    t0 = time.time()
    with open(f"{Z}/finetune.log", "w") as log:
        r = subprocess.run([f"{PYBIN}/microbleednet", "fine_tune", "-i", f"{F}/train", "-l", f"{F}/train/labels", "-m", WEIGHTS, "-o", f"{Z}/modell",
                            "-ep", str(a.epochs), "-es", "10", "-psize", "32", "-bs", "4", "-ftlayers", *map(str, range(1, 9)), "-cp_type", "last", "-sv_mod", "True", "-v", "True"],
                           stdout=log, stderr=subprocess.STDOUT, env=env)
    t1 = time.time()
    if r.returncode != 0:
        raise SystemExit(f"fine_tune rc={r.returncode}, see {Z}/finetune.log")
    with open(f"{Z}/evaluate.log", "w") as log:
        r = subprocess.run([f"{PYBIN}/microbleednet", "evaluate", "-i", f"{F}/test", "-m", f"{Z}/modell/microbleednet", "-o", f"{Z}/auswertung", "-int", "True", "-v", "True"],
                           stdout=log, stderr=subprocess.STDOUT, env=env)
    t2 = time.time()
    if r.returncode != 0:
        raise SystemExit(f"evaluate rc={r.returncode}, see {Z}/evaluate.log")
    preds = glob.glob(f"{Z}/auswertung/final_predictions/predicted_final_microbleednet_*.nii.gz")
    ep = len(re.findall(r"^Epoch: \d+", open(f"{Z}/finetune.log").read(), re.M))
    json.dump(dict(dauer_finetune_s=round(t1 - t0), dauer_evaluate_s=round(t2 - t1), epochen_log=ep, n_pred=len(preds),
                   parameter=dict(ep=a.epochs, es=10, psize=32, bs=4, ftlayers="1-8", ilr=1e-4, weights=WEIGHTS)), open(f"{Z}/lauf.json", "w"), indent=1)
    print(open(f"{Z}/lauf.json").read())


def score(a):
    """Native lesion metric (26-connectivity, touch rule, as mb_metrik) AND our grid metric: the native prediction is resampled
    (nearest) onto our training grid and written as fold<k>/<id>_pred_proba.nii.gz (255 = predicted) so cmb.transfer.evaluate
    and operating_point score it with the project's fixed cell like every other run."""
    from scipy import ndimage
    from nibabel.processing import resample_from_to
    cases = cases_of(a.cohort); root = root_of(a.cohort); N26 = np.ones((3, 3, 3), bool)
    tp = fp = fn = 0; n_pred_cases = 0; out_run = f"{root}/als-lauf"
    for c in cases:
        p = f"{root}/ergebnisse/fold{c['fold']}/auswertung/final_predictions/predicted_final_microbleednet_{c['id']}.nii.gz"
        if not os.path.exists(p):
            continue
        n_pred_cases += 1
        pi = nib.load(p); pred = np.asarray(pi.dataobj) > 0.5
        labfile = f"{root}/bilder/{c['id']}_label.nii.gz"
        if not os.path.exists(labfile):                      # inputs linked from an earlier official preprocessing
            labfile = f"{root}/preproc/labels/{c['id']}_label.nii.gz"
        lab = np.asarray(nib.load(labfile).dataobj) > 0
        rl, nr = ndimage.label(lab, structure=N26); pl, npd = ndimage.label(pred, structure=N26)
        both = lab & pred; hit_r = np.zeros(nr + 1, bool); hit_p = np.zeros(npd + 1, bool)
        hit_r[np.unique(rl[both])] = True; hit_p[np.unique(pl[both])] = True; hit_r[0] = hit_p[0] = False
        tp += hit_r.sum(); fn += nr - hit_r.sum(); fp += npd - hit_p.sum()
        gi = nib.load(f"{c['grid_dir']}/{c['grid_id']}_image.nii.gz")
        rs = resample_from_to(nib.Nifti1Image(pred.astype(np.uint8), pi.affine), (gi.shape, gi.affine), order=0)
        d = f"{out_run}/fold{c['fold']}"; os.makedirs(d, exist_ok=True)
        nib.save(nib.Nifti1Image((np.asarray(rs.dataobj) > 0).astype(np.uint8) * 255, gi.affine), f"{d}/{c['grid_id']}_pred_proba.nii.gz")
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    res = dict(cohort=a.cohort, cases_predicted=n_pred_cases, cases=len(cases), tp=int(tp), fp=int(fp), fn=int(fn), f1=round(f1, 4),
               precision=round(tp / max(tp + fp, 1), 4), sensitivity=round(tp / max(tp + fn, 1), 4), fp_per_case=round(fp / max(n_pred_cases, 1), 2), grid_run=out_run)
    json.dump(res, open(f"{root}/zusammenfassung.json", "w"), indent=1); print(json.dumps(res))
    print(f"our grid metric: python -m cmb.transfer.evaluate --run mbnet-ft={out_run} ... --manifest <manifest of {a.cohort}>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "finetune", "score"]); ap.add_argument("--cohort", required=True, choices=["valdo", "t2star", "swi"])
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--processes", type=int, default=6); ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    {"prepare": prepare, "finetune": finetune, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
