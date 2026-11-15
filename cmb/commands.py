"""The ``cmb`` commands (same names and letters as cmb_3d_unet_experiments / MicrobleedNet), as thin wrappers over the
paper code. Each wrapper only builds the inputs (a manifest of grid folders) and then calls the module that produced the
paper's numbers, with the paper's defaults:

    cmb preprocess     -i RAW -ir _T2starw [-f True] [-b MASKS -br _brainmask.nii.gz] [-l LABELS -lr _cmb.nii.gz] -o GRID
    cmb train          -i GRID [-l GRID] -m MODEL                     cmb.training.full_data train (all cases, seeds 42 + 1)
    cmb fine_tune      -i GRID [-l GRID] -m pre|MODEL -o NEW          the Phase 2 FT / FTn recipe (scripts/paper/phase2_finetune/run_block.py)
    cmb fine_tune      -i GRID -m pre -o NEW --calibrate_only True    the Phase 2 calibration-only arm (calibrate_n0.py): threshold
                                                                      and minimum size re-chosen on the given labelled scans
    cmb evaluate       -i GRID -m MODEL -o PRED                       stage-1 maps + fixed-cell masks (+ lesion F1 when labels exist)
    cmb cross_validate -i GRID [-l GRID] -o CV -fold 5 [--stage2 joint|separate|none]
                                                                      code/turnier.py (frozen folds of the manifest), then
                                                                      cmb.stage2.candidates + cmb.stage2.pooled (tset-plain)

GRID = the folder written by ``cmb preprocess``: <id>/<id>_{image,frst,label}.nii.gz on the 0.5 x 0.5 x 1 mm grid plus
manifest.json (id, fold, dir, cohort, n_cmb). A model folder holds <member>/modell.pt + meta.json.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from cmb.paths import ROOT, PY

TRUE = ("true", "1", "yes", "y", "t")


def as_bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in TRUE


def _run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc {r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


# ---------------------------------------------------------------------------------------------------- manifests
def find(folder: str, case: str, suffix: str):
    for cand in (f"{case}{suffix}", f"{case}{suffix}.nii.gz", f"{case}{suffix}.nii"):
        p = os.path.join(folder, cand)
        if os.path.exists(p):
            return p
    return None


def assign_folds(entries, n_folds: int):
    """Stratified by CMB count: sort by (n_cmb, id), deal round robin. Deterministic, documented in the README."""
    for k, e in enumerate(sorted(entries, key=lambda e: (e["n_cmb"], e["id"]))):
        e["fold"] = k % n_folds
    return entries


def grid_manifest(grid: str, labels: str = None, n_folds: int = 5) -> str:
    """Absolute-path manifest for the grid folder (reuses <grid>/manifest.json folds). With -l != -i the images come
    from -i and the labels from -l (a merged folder of symlinks)."""
    grid = os.path.abspath(grid)
    src = os.path.join(grid, "manifest.json")
    if os.path.exists(src):
        entries = json.load(open(src))
    else:
        ids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{grid}/*/*_image.nii.gz"))
        if not ids:
            raise SystemExit(f"{grid}: no <id>/<id>_image.nii.gz -- run cmb preprocess first")
        entries = assign_folds([dict(id=i, n_cmb=0, cohort="data") for i in ids], n_folds)
    if labels and os.path.abspath(labels) != grid:
        merged = os.path.join(grid, ".merged_labels")
        for e in entries:
            d = os.path.join(merged, e["id"]); os.makedirs(d, exist_ok=True)
            for kind, base in (("image", grid), ("frst", grid), ("label", os.path.abspath(labels))):
                t = os.path.join(d, f"{e['id']}_{kind}.nii.gz")
                if not os.path.lexists(t):
                    os.symlink(os.path.join(base, e["id"], f"{e['id']}_{kind}.nii.gz"), t)
        base_dir = merged
    else:
        base_dir = grid
    for e in entries:
        e["dir"] = os.path.join(base_dir, e["id"])
    out = os.path.join(grid, ".manifest_abs.json")
    json.dump(entries, open(out, "w"), indent=1)
    return out


# ---------------------------------------------------------------------------------------------------- preprocess
def preprocess(a) -> None:
    """Raw scans -> the paper's model grid (brain mask, FSL FAST -B, inversion, 0.5 x 0.5 x 1 mm, FRST)."""
    import nibabel as nib
    from scipy import ndimage
    from cmb import predict_cli as P
    raw = a.i
    files = sorted(f for f in glob.glob(os.path.join(raw, f"*{a.ir}*")) if f.endswith((".nii", ".nii.gz")))
    if not files:
        raise SystemExit(f"no files *{a.ir}*.nii[.gz] in {raw}")
    out = os.path.abspath(a.o); os.makedirs(out, exist_ok=True)
    entries = []
    for f in files:
        case = os.path.basename(f).split(a.ir)[0]
        t0 = time.time()
        ni = nib.load(f); img = np.nan_to_num(np.asarray(ni.dataobj, dtype=np.float32))
        work = os.path.join(out, f".work_{case}"); os.makedirs(work, exist_ok=True)
        mask = find(a.b, case, a.br) if a.b else None
        if a.b and not mask:
            raise SystemExit(f"{case}: no brain mask {case}{a.br} in {a.b}")
        brain = P.brain_mask(img, ni, mask, work, lambda m: None)
        masked = np.where(brain, img, 0).astype(np.float32)
        restore = masked if as_bool(a.f) else P.bias_correct(masked, ni, "auto", work, lambda m: None)
        inverted = P.invert(restore)
        lab = np.zeros(img.shape, np.uint8)
        if a.l:
            lp = find(a.l, case, a.lr)
            if not lp:
                raise SystemExit(f"{case}: no label {case}{a.lr} in {a.l}")
            lab = (np.asarray(nib.load(lp).dataobj) > 0).astype(np.uint8)
        src = os.path.join(work, "source", case); os.makedirs(src, exist_ok=True)
        nib.save(nib.Nifti1Image(inverted, ni.affine, ni.header), f"{src}/{case}_image.nii.gz")
        h8 = ni.header.copy(); h8.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(lab, ni.affine, h8), f"{src}/{case}_label.nii.gz")
        sys.path.insert(0, os.path.join(ROOT, "code"))
        import gitter_c_bauen as builder
        builder.QUELLE, builder.ZIEL = os.path.dirname(src), out
        info = builder.fall((case, 0, False))
        if "fehler" in info:
            raise SystemExit(f"{case}: grid builder failed: {info['fehler']}")
        shutil.rmtree(work, ignore_errors=True)
        entries.append(dict(id=case, n_cmb=int(info["n_cmb"]), cohort=a.cohort))
        print(f"{case}: grid {info['shape']}, {info['n_cmb']} CMB [{time.time() - t0:.0f} s]", flush=True)
    assign_folds(entries, a.fold)
    json.dump([dict(id=e["id"], fold=e["fold"], dir=e["id"], cohort=e["cohort"], n_cmb=e["n_cmb"]) for e in entries],
              open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"{len(entries)} cases -> {out} (manifest.json with {a.fold} folds)")


# ---------------------------------------------------------------------------------------------------- train
def train(a) -> None:
    man = grid_manifest(a.i, a.l)
    for seed in a.seeds:
        cmd = [PY, "-m", "cmb.training.full_data", "train", "--manifest", man, "--out", os.path.join(os.path.abspath(a.m), f"seed{seed}"),
               "--seed", seed, "--epochs", a.ep]
        if not as_bool(a.frst):
            cmd += ["--channels", "t2s"]
        if a.ilr is not None:
            cmd += ["--lr", a.ilr]
        if a.bs is not None:
            cmd += ["--bs", a.bs]
        if a.patches:
            cmd += ["--patches", a.patches]
        if a.overwrite:
            cmd += ["--force"]
        _run(cmd, env=dict(os.environ, CV5_DEV=_device(a.device)))
    print(f"model -> {a.m} ({len(a.seeds)} member(s))")


def _device(d: str) -> str:
    if d != "auto":
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- fine_tune
def freeze_for_ftn(model):
    """scripts/paper/phase2_finetune/run_block.py: keep only InstanceNorm affine parameters and the output head trainable
    (Karani et al. 2018)."""
    import torch.nn as nn
    n_trainable = 0
    for mname, m in model.named_modules():
        for pname, p in m.named_parameters(recurse=False):
            full = f"{mname}.{pname}" if mname else pname
            if isinstance(m, (nn.InstanceNorm3d, nn.InstanceNorm2d, nn.InstanceNorm1d)) or ".head" in full:
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
    return n_trainable


def start_model(m: str) -> str:
    if m in ("pre", "pretrained", "valdo_t2"):
        from cmb.pretrained import get_model
        return get_model("valdo_t2")
    return m


def fine_tune(a) -> None:
    if as_bool(a.calibrate_only):
        return calibrate(a)
    import torch
    os.environ["CV5_DEV"] = _device(a.device)
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import cv5
    import turnier
    cv5.DEV = os.environ["CV5_DEV"]
    man = grid_manifest(a.i, a.l)
    turnier.mit_manifest([man])()
    ids = sorted(cv5.FA["recs"])
    src = start_model(a.m)
    members = sorted(glob.glob(os.path.join(src, "*", "modell.pt")))
    if not members:
        raise SystemExit(f"no <member>/modell.pt in {src}")
    variant = "FTn" if a.ft_layers == "norm_out" else "FT"
    epochs = a.ep or 30                       # Phase 2: half the 60-epoch recipe
    lr = a.ilr or 1e-4                        # Phase 2: 1e-4 (the recipe's 3e-4 for training from scratch)
    for path in members:
        meta0 = json.load(open(os.path.join(os.path.dirname(path), "meta.json")))
        seed = int(meta0.get("seed", 42))
        member = os.path.basename(os.path.dirname(path))
        out = os.path.join(os.path.abspath(a.o), member); os.makedirs(out, exist_ok=True)
        t0 = time.time()
        torch.manual_seed(seed); rng = np.random.RandomState(seed)
        P = dict(meta0["params"]); P["loss"] = "bce_dice_ds"
        if a.patches:
            P["n_patches"] = a.patches
        if a.bs:
            P["bs"] = a.bs
        model, nch = cv5.neues_modell(P)
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        if variant == "FTn":
            freeze_for_ftn(model)
        model.to(cv5.DEV)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = cv5.VERLUSTE["bce_dice_ds"]
        history = []
        for ep in range(1, epochs + 1):
            model.train(); losses = []
            xb, yb = cv5.stichprobe(ids, rng, P)
            for b0 in range(0, len(xb), P["bs"]):
                x2, y2 = xb[b0:b0 + P["bs"]].to(cv5.DEV), yb[b0:b0 + P["bs"]].to(cv5.DEV)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cv5.DEV == "cuda"):
                    loss = loss_fn(model(x2), y2)
                loss.backward(); opt.step(); losses.append(float(loss))
            sched.step()
            history.append(dict(epoch=ep, loss=round(float(np.mean(losses)), 4)))
            print(f"[{variant} {member}] ep {ep}/{epochs} loss {history[-1]['loss']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        torch.save(model.state_dict(), os.path.join(out, "modell.pt"))
        json.dump(history, open(os.path.join(out, "verlauf.json"), "w"), indent=1)
        meta = dict(ok=True, net=meta0.get("net", "a03-aniso"), seed=seed, params=P, nch=nch, epochs=epochs, lr=lr,
                    variant=variant, start=os.path.abspath(path), final_state=True, n_train=len(ids), cases=ids,
                    n_trainable_params=int(sum(p.numel() for p in params)))
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"fine-tuned model ({variant}, {epochs} epochs, lr {lr}) -> {a.o}")


def calibrate(a) -> None:
    """Calibration-only arm (scripts/paper/phase2_finetune/calibrate_n0.py): the model is unchanged; threshold and
    minimum size are chosen to maximise pooled lesion F1 on the given labelled scans. Writes <o>/calibration.json and
    a copy of the model folder with the new cell in model.json (cmb evaluate / predict read it)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts", "paper", "phase2_finetune"))
    import nibabel as nib
    import calibrate_n0 as C
    src = start_model(a.m)
    pred = os.path.join(os.path.abspath(a.o), "pred_on_calibration_scans")
    ev = type("A", (), dict(i=a.i, m=src, o=pred, l=a.l, device=a.device, csf_rule=False))
    evaluate(ev)
    entries = json.load(open(grid_manifest(a.i, a.l)))
    tab = None
    for e in entries:
        prob = np.asarray(nib.load(f"{pred}/fold0/{e['id']}_pred_proba.nii.gz").dataobj).astype(np.float32) / 255.0
        ref = np.asarray(nib.load(f"{e['dir']}/{e['id']}_label.nii.gz").dataobj) > 0
        t = C.scan_table(prob, ref); tab = t if tab is None else tab + t
    f, *_ = C.f1(tab)
    i03, j2 = int(np.argmin(abs(C.TS - 0.3))), C.MINS.index(2.0)
    if np.all(np.isnan(f)):
        i, j = i03, j2
    else:
        i, j = np.unravel_index(np.nanargmax(f), f.shape)
    cell = dict(threshold=float(C.TS[i]), min_mm3=float(C.MINS[j]), max_voxels=2100)
    out = os.path.abspath(a.o)
    for d in glob.glob(os.path.join(src, "*", "")):
        shutil.copytree(d, os.path.join(out, os.path.basename(os.path.dirname(d))), dirs_exist_ok=True)
    info = json.load(open(os.path.join(src, "model.json"))) if os.path.exists(os.path.join(src, "model.json")) else {}
    info.update(cell=cell, calibrated_on=[e["id"] for e in entries], calibration="calibrate_only (Phase 2 CAL arm)")
    json.dump(info, open(os.path.join(out, "model.json"), "w"), indent=1)
    json.dump(dict(cell=cell, f1_on_calibration_scans=float(f[i, j]) if not np.isnan(f[i, j]) else None,
                   fixed_cell_f1=float(f[i03, j2]) if not np.isnan(f[i03, j2]) else None, n_scans=len(entries)),
              open(os.path.join(out, "calibration.json"), "w"), indent=1)
    print(f"calibrated cell: threshold {cell['threshold']}, min {cell['min_mm3']} mm3 -> {out}/model.json")


# ---------------------------------------------------------------------------------------------------- evaluate
def model_cell(model_dir: str) -> dict:
    p = os.path.join(model_dir, "model.json")
    cell = json.load(open(p)).get("cell") if os.path.exists(p) else None
    return cell or dict(threshold=0.3, min_mm3=2.0, max_voxels=2100)


def evaluate(a) -> None:
    """Stage-1 maps (mean of the members) -> <o>/fold0/<id>_pred_proba.nii.gz (uint8 * 255, the paper's layout),
    masks with the model's cell -> <o>/masks/<id>_cmb_mask.nii.gz, and lesion F1 when the grid has labels."""
    import nibabel as nib
    from scipy import ndimage
    model = start_model(a.m)
    man = grid_manifest(a.i, getattr(a, "l", None))
    pred = os.path.abspath(a.o)
    entries = json.load(open(man))
    flat = os.path.join(pred, ".manifest_fold0.json")
    os.makedirs(pred, exist_ok=True)
    json.dump([dict(e, fold=0) for e in entries], open(flat, "w"), indent=1)
    members = sorted(glob.glob(os.path.join(model, "*", "modell.pt")))
    _run([PY, "-m", "cmb.training.full_data", "predict", "--models", *members, "--manifest", flat, "--out", pred, "--force"],
         env=dict(os.environ, CV5_DEV=_device(a.device)))
    cell = model_cell(model)
    sys.path.insert(0, os.path.join(ROOT, "code"))
    from metric import hits, summarize
    os.makedirs(os.path.join(pred, "masks"), exist_ok=True)
    rows = []
    for e in entries:
        ni = nib.load(f"{pred}/fold0/{e['id']}_pred_proba.nii.gz")
        p = np.asarray(ni.dataobj).astype(np.float32) / 255.0
        lab, n = ndimage.label(p > cell["threshold"], structure=np.ones((3, 3, 3), bool))
        size = np.bincount(lab.ravel(), minlength=n + 1)
        keep = (size * 0.25 >= cell["min_mm3"]) & (size <= cell["max_voxels"]); keep[0] = False
        m = keep[lab]
        out_lab, k = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
        nib.save(nib.Nifti1Image(out_lab.astype(np.int16), ni.affine), f"{pred}/masks/{e['id']}_cmb_mask.nii.gz")
        lp = f"{e['dir']}/{e['id']}_label.nii.gz"
        row = dict(id=e["id"], n_detected=int(k))
        if os.path.exists(lp):
            ref = np.asarray(nib.load(lp).dataobj) > 0
            if ref.any() or e.get("n_cmb", 0) > 0:
                h = hits(ref, m); row.update(h); rows.append(h)
        print(f"{e['id']}: {k} CMB", flush=True)
    summary = dict(cell=cell, model=model)
    if rows:
        s = summarize(rows); summary.update({k: round(float(v), 4) for k, v in s.items() if isinstance(v, (int, float))})
        print(f"lesion F1 {s['f1']:.3f} (precision {s['precision']:.2f}, sensitivity {s['sensitivity']:.2f}) on {len(rows)} scans with labels")
    json.dump(summary, open(os.path.join(pred, "summary.json"), "w"), indent=1)


def _candidates(run: str, manifest: str, out: str) -> bool:
    """cmb.stage2.candidates with the paper's z40 options (as run_block.py). False (with a warning) when stage 1 proposed
    no candidate at all -- e.g. a barely trained model -- because then there is nothing for stage 2 to learn."""
    cmd = [PY, "-m", "cmb.stage2.candidates", "--predictions", run, "--manifest", manifest, "--out", out, "--threshold", "0.15",
           "--min-mm3", "1.0", "--half", "20", "20", "20", "--keep-csf", "--csf-mode", "majority", "--synthseg", "none"]
    print("  $ " + " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:], end="", flush=True)
    if r.returncode != 0 and "need at least one array" in r.stderr:
        print("WARNING: stage 2 skipped: stage 1 proposed no candidates (probability > 0.15, >= 1 mm3) in "
              f"{manifest} -- train stage 1 longer", file=sys.stderr, flush=True)
        return False
    if r.returncode != 0:
        raise SystemExit(f"cmb.stage2.candidates failed:\n{r.stderr[-2000:]}")
    return True


# ---------------------------------------------------------------------------------------------------- cross_validate
def cross_validate(a) -> None:
    """code/turnier.py over the folds of the manifest (inner-fold checkpoint selection, out-of-fold maps), the fixed-cell
    score (cmb.transfer.evaluate), then optionally stage 2 on the out-of-fold candidates (tset-plain, cmb.stage2.pooled):
    joint = one classifier for all cohorts of the manifest ("pooled"), separate = one per cohort ("own")."""
    if a.fold != 5:
        raise SystemExit("-fold: the paper's CV code (code/turnier.py, cv5.train_falte) uses 5 folds (inner fold = k+1 mod 5)")
    man = grid_manifest(a.i, a.l, a.fold)
    entries = json.load(open(man))
    if {e["fold"] for e in entries} != set(range(a.fold)):      # e.g. a grid prepared with another -fold
        assign_folds(entries, a.fold)
        json.dump(entries, open(man, "w"), indent=1)
        print(f"folds re-assigned to {a.fold} (stratified by CMB count)", flush=True)
    out = os.path.abspath(a.o); os.makedirs(out, exist_ok=True)
    name = "cv"
    cmd = [PY, os.path.join(ROOT, "code", "turnier.py"), "--netz", "a03-aniso", "--manifest", man, "--name", name,
           "--basis", out, "--seed", a.seed, "--falten", *range(a.fold)]
    if a.ep:
        cmd += ["--epochen", a.ep]
    if not as_bool(a.frst):
        cmd += ["--kanaele", "t2s"]
    if a.ilr:
        cmd += ["--lr", a.ilr]
    hp = []
    if a.patches:
        hp.append(f"n_patches={a.patches}")
    if a.bs:
        hp.append(f"bs={a.bs}")
    if hp:
        cmd += ["--hp", *hp]
    _run(cmd, env=dict(os.environ, CV5_DEV=_device(a.device)))
    runs = sorted(d for d in glob.glob(os.path.join(out, "*")) if os.path.isdir(os.path.join(d, "fold0")))
    if not runs:
        raise SystemExit(f"no run folder with fold0/ under {out}")
    run = runs[-1]
    _run([PY, "-m", "cmb.transfer.evaluate", "--run", f"cv={run}", "--manifest", man, "--json", os.path.join(out, "summary.json")])
    if a.stage2 != "none":
        cand = os.path.join(out, "stage2_candidates")
        if not _candidates(run, man, cand):
            return
        cohorts = sorted({e.get("cohort", "data") for e in entries})
        specs = []
        if len(cohorts) == 1:
            specs = [f"{cohorts[0]}={cand}"]
        else:
            for c in cohorts:          # one candidate folder per cohort (cmb.stage2.pooled pools them by name)
                sub = [e for e in entries if e.get("cohort", "data") == c]
                m2 = os.path.join(out, f".manifest_{c}.json"); json.dump(sub, open(m2, "w"))
                d = os.path.join(out, f"stage2_candidates_{c}")
                if not _candidates(run, m2, d):
                    return
                specs.append(f"{c}={d}")
        cmd = [PY, "-m", "cmb.stage2.pooled", "--cohort", *specs, "--models", "tset-plain", "--channels", "all",
               "--seed-offset", "0", "--json", os.path.join(out, "stage2.json")]
        if a.stage2 == "joint":
            cmd.append("--skip-own")
        _run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES="" if _device(a.device) == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    print(f"cross-validation -> {out}/summary.json" + (f", stage 2 -> {out}/stage2.json" if a.stage2 != "none" else ""))
