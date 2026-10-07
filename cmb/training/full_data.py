"""Train the anisotropic U-Net on ALL cases of one or more manifests -- no folds, no inner validation -- and predict with it.

Why this exists (user request 2026-10-07): the deliverable models of the product package
(``microbleed_detection_3D_UNET``, ``mb_segment -m valdo_t2 | charite_t2 | charite_swi``) must see every
annotated case, not four fifths of them. The cross-validation runs (``code/turnier.py``) keep one fold for the
score and one for the checkpoint; here nothing is held back, so there is no checkpoint selection either:
the state after the LAST epoch of the fixed 60-epoch cosine schedule is kept. That choice was measured on the
CV runs: last state instead of best inner-fold state -0.023 [-0.069; +0.012], within noise (WORKLOG 5o).

Everything else is the recipe of the CV runs, byte for byte the same code paths: ``cv5.stichprobe`` (800 patches
per epoch, 60 % lesion / 25 % brain / 15 % anywhere, flips), ``cv5.VERLUSTE['bce_dice_ds']`` (BCE + Dice with
deep supervision), AdamW 3e-4 / 1e-4, bfloat16, ``turnier.NETZE['a03-aniso']``. The operating point for the
product is the project's fixed cell (threshold 0.3, >= 2 mm3, <= 2100 voxels), never re-tuned.

    python -m cmb.training.full_data train   --manifest M [M ...] --out DIR --seed 42 [--epochs 60] [--patches 800]
    python -m cmb.training.full_data predict --models DIR1/modell.pt [DIR2/modell.pt ...] --manifest M [M ...] --out RUN

``predict`` averages the given models (the two-seed ensemble of the deliverable) and writes
``RUN/fold<k>/<id>_pred_proba.nii.gz`` (uint8, probability * 255; k = the case's fold from the manifest), the
layout that ``cmb.transfer.evaluate`` and ``cmb.analysis.operating_point`` score. Used for the cross tests
(VALDO model on the in-house cohorts, in-house models on VALDO).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
import cv5      # noqa: E402
import turnier  # noqa: E402  -- registers the networks and the deep-supervision loss in cv5


def absolute(path: str) -> str:
    return path if os.path.isabs(path) else f"{ROOT}/{path}"


def load_manifests(paths):
    loader = turnier.mit_manifest([absolute(p) for p in paths])
    loader()
    return cv5.FA["recs"]


def train(a) -> None:
    cv5.SEED = a.seed
    recs = load_manifests(a.manifest)
    train_ids = sorted(recs)
    P = cv5.konf_params(a.net)
    if a.patches:
        P["n_patches"] = a.patches
    out = absolute(a.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/meta.json") and json.load(open(f"{out}/meta.json")).get("ok") and not a.force:
        print(f"{out}: finished -- skipped (--force repeats)", flush=True)
        return
    t0 = time.time()
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    model, nch = cv5.neues_modell(P)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = cv5.VERLUSTE[P["loss"]]
    if cv5.DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = []
    bs = P["bs"]
    for ep in range(1, a.epochs + 1):
        model.train()
        t_ep = time.time()
        losses = []
        xb, yb = cv5.stichprobe(train_ids, rng, P)
        for b0 in range(0, len(xb), bs):
            x2, y2 = xb[b0:b0 + bs].to(cv5.DEV), yb[b0:b0 + bs].to(cv5.DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cv5.DEV == "cuda"):
                loss = loss_fn(model(x2), y2)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        history.append(dict(epoch=ep, loss=round(float(np.mean(losses)), 4), train_s=round(time.time() - t_ep, 1)))
        print(f"[{a.net} all {len(train_ids)} cases, seed {a.seed}] ep {ep:2d}/{a.epochs} loss {history[-1]['loss']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    torch.save(model.state_dict(), f"{out}/modell.pt")
    json.dump(history, open(f"{out}/verlauf.json", "w"), indent=1)
    meta = dict(ok=True, net=a.net, seed=a.seed, params=P, nch=nch, epochs=a.epochs, final_state=True,
                note="trained on ALL cases of the manifests; no inner fold, the state after the last epoch is kept",
                manifests=[absolute(p) for p in a.manifest], n_train=len(train_ids), cases=train_ids,
                n_cmb=int(sum(len(r["komp"]) for r in recs.values())),
                fixed_cell=dict(threshold=0.3, min_mm3=2.0, max_voxels=cv5.MAX_KOMP_VOX),
                duration_s=round(time.time() - t0, 1),
                vram_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if cv5.DEV == "cuda" else None,
                torch=torch.__version__, parameters=turnier._LETZTES.get("parameter"))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"FINISHED {out}: {len(train_ids)} cases, {meta['n_cmb']} CMB, {a.epochs} epochs, {meta['duration_s']:.0f} s, "
          f"VRAM {meta['vram_gib']} GiB", flush=True)


def load_models(paths):
    models = []
    for p in paths:
        p = absolute(p)
        meta = json.load(open(os.path.join(os.path.dirname(p), "meta.json")))
        P = meta["params"]
        model, nch = cv5.neues_modell(P)
        model.load_state_dict(torch.load(p, map_location=cv5.DEV))
        model.to(cv5.DEV).eval()
        models.append((model, nch, p))
    return models


def predict(a) -> None:
    recs = load_manifests(a.manifest)
    models = load_models(a.models)
    out = absolute(a.out)
    back = lambda arr: np.ascontiguousarray(arr.transpose(2, 1, 0))      # (z, y, x) -> (x, y, z)
    t0 = time.time()
    done = 0
    for sid in sorted(recs):
        r = recs[sid]
        folder = f"{out}/fold{r['fold']}"
        os.makedirs(folder, exist_ok=True)
        target = f"{folder}/{sid}_pred_proba.nii.gz"
        if os.path.exists(target) and not a.force:
            continue
        p = np.mean([cv5.proba_fall(m, sid, nch=nch)[0] for m, nch, _ in models], axis=0)
        nib.save(nib.Nifti1Image(back(np.round(p * 255).astype(np.uint8)), r["affine"]), target)
        done += 1
        print(f"{sid}: fold {r['fold']}, max p {p.max():.2f}  [{time.time() - t0:.0f}s]", flush=True)
    json.dump(dict(models=[p for _, _, p in models], manifests=[absolute(p) for p in a.manifest], cases=sorted(recs),
                   note="mean of the listed models; uint8 = probability * 255"),
              open(f"{out}/predict_meta.json", "w"), indent=1)
    print(f"FINISHED {out}: {done} new maps, {len(recs)} cases, {len(models)} models", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--manifest", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--net", default="a03-aniso")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patches", type=int, default=None, help="patches per epoch (recipe: 800); small values only for smoke tests")
    t.add_argument("--force", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--models", nargs="+", required=True, help="modell.pt files; meta.json must sit next to each")
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (train if a.cmd == "train" else predict)(a)


if __name__ == "__main__":
    main()
