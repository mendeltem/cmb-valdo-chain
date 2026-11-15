"""Phase-2 fine-tuning runs: one block = one (modality, fold k, variant, n, draw seed).

Trains the a03-aniso U-Net on the nested draw of n training scans of fold k, predicts the
frozen test fold k, builds z40 candidates for the test scans (for the stage-2 chain), and
scores stage 1 with the fixed post-processing (0.3 / 2 mm3). Stage-2 chains (frozen-V /
updated / own-D) are scored by phase4_stage2.py once the candidate sets exist.

Variants
  S     from scratch on D's n scans, 60 epochs, lr 3e-4 (arena recipe, no inner selection:
        the state after the LAST epoch, as in cmb.training.full_data)
  FT    the VALDO-72 U-Net (full-data ensemble members, one per seed) fully fine-tuned on
        D's n scans, 30 epochs (half), lr 1e-4 (half of the recipe's 3e-4)
  FTn   like FT but only the InstanceNorm affine parameters and the output head are
        updated (Karani 2018); only offered for n = 5, 10, 20

    python scripts/paper/phase2_finetune/run_block.py --mod t2star --fold 0 --variant FT --n 5 --seed 42
    python scripts/paper/phase2_finetune/run_block.py --mod t2star --fold 0 --variant S --n all --seed 42
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
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import nibabel as nib

ROOT = _R
MB = _P
PAPER = f"{MB}/paper/phase4"
sys.path.insert(0, f"{ROOT}/code")
sys.path.insert(0, ROOT)
import cv5      # noqa: E402
import turnier  # noqa: E402
from netze import NETZE                 # noqa: E402

NS = [3, 5, 10, 15, 20, 30, 40]
# 31.10.2026: SWI reads the hole-filled grids of wk0a (gitter/swi_v2; same ids and frozen folds). T2* keeps
# gitter/t2star: the fill changes no T2* reference lesion (4/828 outside the mask before and after), and fold 0 ran on it.
GRID = {"t2star": f"{MB}/gitter/manifest_t2star.json", "swi": f"{MB}/gitter/manifest_swi_v2.json"}
STRATEGIES = ("STRAT", "RAND", "HIGH", "LOW", "MID", "NONE")


def block_name(variant, n, fold, seed, strategy="STRAT"):
    """Output folder name; STRAT keeps the old names so finished blocks are found again."""
    return f"{variant}_n{n}_fold{fold}_s{seed}" + ("" if strategy == "STRAT" else f"_{strategy}")
# The VALDO full-data ensemble has two members (seed42, seed1). Block seed 7 re-uses the
# seed42 weights (init only; the fine-tuning seed is still 7).
VALDO_MEMBERS = {42: f"{ROOT}/ergebnisse/final/valdo_t2/seed42/modell.pt",
                 1: f"{ROOT}/ergebnisse/final/valdo_t2/seed1/modell.pt",
                 7: f"{ROOT}/ergebnisse/final/valdo_t2/seed42/modell.pt"}


def manifest_of(mod, fold, variant, n, seed, strategy="STRAT"):
    """The training cases: the nested draw prefix (or ALL training scans for n = all)."""
    if n == "all":
        entries = json.load(open(GRID[mod]))
        ids = [e["id"] for e in entries if e["fold"] != fold]
        src = "all training folds (folds_frozen)"
    else:
        d = json.load(open(f"{PAPER}/draws/{mod}/fold{fold}_n{n}_{strategy}_{seed}.json"))
        ids = d["cases"]
        src = d["source_manifest"]
    return ids, src


def freeze_for_ftn(model):
    """Keep only InstanceNorm affine parameters (weight/bias) and the output head trainable
    (Karani et al. 2018, DOI 10.1007/978-3-030-00928-1_54)."""
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


def build_model(variant, seed):
    """The arena winner a03-aniso (code/netze.py, Polster wrapper) -- same object the
    full-data checkpoints were trained with."""
    nch = 2
    m = NETZE["a03-aniso"](nch)
    if variant in ("FT", "FTn"):
        m.load_state_dict(torch.load(VALDO_MEMBERS[seed], map_location="cpu"), strict=True)
        if variant == "FTn":
            meta_ntrain = freeze_for_ftn(m)
    return m.to(cv5.DEV), nch


def run(a):
    ids, src = manifest_of(a.mod, a.fold, a.variant, a.n, a.seed, a.strategy)
    # Load the grids of ALL scans of this modality once (prediction needs the test scans;
    # sampling needs the training scans).
    loader = turnier.mit_manifest([GRID[a.mod]])
    loader()
    out = f"{PAPER}/unet/{a.mod}/{block_name(a.variant, a.n, a.fold, a.seed, a.strategy)}"
    if a.force:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/meta.json") and json.load(open(f"{out}/meta.json")).get("ok"):
        print(f"{out}: already done -- skipped", flush=True)
        return
    t0 = time.time()
    epochs = a.epochs or (30 if a.variant.startswith("FT") else 60)
    lr = a.lr or (1e-4 if a.variant.startswith("FT") else 3e-4)
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    model, nch = build_model(a.variant, a.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = cv5.VERLUSTE["bce_dice_ds"]   # arena a03-aniso recipe (turnier.py registers it)
    P = dict(cv5.konf_params("a03-aniso"))
    P["loss"] = "bce_dice_ds"
    if a.patches:
        P["n_patches"] = a.patches
    if cv5.DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t_ep = time.time(); losses = []
        xb, yb = cv5.stichprobe(ids, rng, P)
        for b0 in range(0, len(xb), P["bs"]):
            x2, y2 = xb[b0:b0 + P["bs"]].to(cv5.DEV), yb[b0:b0 + P["bs"]].to(cv5.DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cv5.DEV == "cuda"):
                loss = loss_fn(model(x2), y2)
            loss.backward(); opt.step(); losses.append(float(loss))
        sched.step()
        history.append(dict(epoch=ep, loss=round(float(np.mean(losses)), 4), train_s=round(time.time() - t_ep, 1)))
        print(f"[{a.variant} n={a.n} f{a.fold} {a.mod}] ep {ep:2d}/{epochs} loss {history[-1]['loss']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    torch.save(model.state_dict(), f"{out}/modell.pt")
    json.dump(history, open(f"{out}/verlauf.json", "w"), indent=1)
    meta = dict(ok=True, net="a03-aniso", mod=a.mod, fold=a.fold, variant=a.variant, n=a.n, seed=a.seed,
                epochs=epochs, lr=lr, final_state=True, cases=ids, case_source=src, strategy=a.strategy, grid=GRID[a.mod],
                n_cmb=int(sum(len(cv5.FA["recs"][i]["komp"]) for i in ids)),
                duration_s=round(time.time() - t0, 1),
                vram_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if cv5.DEV == "cuda" else None,
                n_trainable_params=int(sum(p.numel() for p in params)))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"TRAIN DONE {out}: {len(ids)} cases, {meta['n_cmb']} CMB, {meta['duration_s']:.0f} s", flush=True)

    # Predict the FROZEN test fold k (out-of-fold: the model never saw these scans).
    entries = json.load(open(GRID[a.mod]))
    test_ids = [e["id"] for e in entries if e["fold"] == a.fold]
    pred_out = f"{out}/pred/fold{a.fold}"
    os.makedirs(pred_out, exist_ok=True)
    model.eval()
    # proba_fall returns the map in internal (z,y,x); the grid files are in the NIfTI
    # orientation, so transpose back, as cmb.transfer.zero_shot does (uint8 * 255).
    t1 = time.time()
    for sid in test_ids:
        target = f"{pred_out}/{sid}_pred_proba.nii.gz"
        if os.path.exists(target) and not a.force:
            continue
        r = cv5.FA["recs"][sid]
        p = cv5.proba_fall(model, sid, nch=nch)[0]
        back = np.ascontiguousarray(p.transpose(2, 1, 0))
        nib.save(nib.Nifti1Image(np.round(back * 255).astype(np.uint8), r["affine"]), target)
    json.dump(dict(models=[f"{out}/modell.pt"], mod=a.mod, fold=a.fold, variant=a.variant, n=a.n, strategy=a.strategy,
                   seed=a.seed, cases=test_ids), open(f"{out}/pred_meta.json", "w"), indent=1)
    print(f"PREDICT DONE {pred_out}: {len(test_ids)} test scans, {time.time() - t1:.0f} s", flush=True)

    # z40 candidates for the test scans (same options as the pool: --keep-csf --half 20 20 20)
    ns = [3, 5, 10, 15, 20, 30, 40]
    nn = "all" if a.n == "all" else (max(ns) if int(a.n) > max(ns) else int(a.n))
    cand_out = f"{PAPER}/candidates/{a.mod}/{block_name(a.variant, a.n, a.fold, a.seed, a.strategy)}"
    if not os.path.exists(f"{cand_out}/candidates.json") or a.force:
        shutil.rmtree(cand_out, ignore_errors=True); os.makedirs(cand_out, exist_ok=True)
        import argparse as _ap, subprocess
        r = subprocess.run([_PY, "-m", "cmb.stage2.candidates",
                            "--predictions", os.path.dirname(pred_out), "--manifest", GRID[a.mod],   # candidates.py globs <dir>/fold*/ (fix 01.11.: pred_out found no files, every set was empty)
                            "--out", cand_out, "--threshold", "0.15", "--min-mm3", "1.0",
                            "--half", "20", "20", "20", "--keep-csf", "--csf-mode", "majority",
                            "--synthseg", "none"], cwd=ROOT, capture_output=True, text=True)
        print(r.stdout[-800:]); print(r.stderr[-800:])
        if r.returncode != 0 and "need at least one array" in r.stderr and "0 candidates" not in r.stdout and len(os.listdir(pred_out)) == 0:
            raise SystemExit(f"candidates found no prediction files under {os.path.dirname(pred_out)}/fold*/")
        if r.returncode != 0 and "need at least one array" in r.stderr:
            # zero candidates above threshold (possible for a barely-trained smoke model):
            # write an empty, well-formed candidate set instead of crashing.
            json.dump(dict(threshold=0.15, min_mm3=1.0, half=[20, 20, 20], tta=None,
                           candidates=[], cases={}, empty=True),
                      open(f"{cand_out}/candidates.json", "w"), indent=1)
            print(f"ZERO CANDIDATES in {cand_out} -- empty set written", flush=True)
        assert os.path.exists(f"{cand_out}/candidates.json"), f"cand.rc={r.returncode}"
    print(f"BLOCK DONE {a.variant} n={a.n} f{a.fold} {a.mod} seed {a.seed}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mod", choices=["t2star", "swi"], required=True)
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument("--variant", choices=["S", "FT", "FTn"], required=True)
    ap.add_argument("--n", required=True, help="3 5 10 15 20 30 40 all")
    ap.add_argument("--seed", type=int, default=42, choices=[42, 1, 7])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patches", type=int, default=None, help="few patches per epoch (smoke test only)")
    ap.add_argument("--strategy", choices=STRATEGIES, default="STRAT", help="draw strategy (RAND/HIGH/LOW: FT at n = 5, 10, 20)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
