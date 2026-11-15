"""n = 0 probability maps for the calibration control (31.10.2026): the VALDO full-data U-Net members (seed42, seed1),
unchanged, applied to EVERY scan of one in-house modality; the two maps are averaged (the "VALDO model as is" of
phase4_eval_*_n0.json). Stored as uint8 * 255 in the NIfTI orientation of the grids, like run_block.py.

    python predict_n0.py --mod t2star        # -> phase4/unet/<mod>/N0_valdo72/pred/all/<id>_pred_proba.nii.gz
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_block as rb          # noqa: E402  (same ROOT, grids, model builder and VALDO members)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mod", choices=["t2star", "swi"], required=True)
    a = ap.parse_args()
    rb.turnier.mit_manifest([rb.GRID[a.mod]])()
    ids = [e["id"] for e in json.load(open(rb.GRID[a.mod]))]
    out = f"{rb.PAPER}/unet/{a.mod}/N0_valdo72"
    pred = f"{out}/pred/all"; os.makedirs(pred, exist_ok=True)
    models = []
    for seed in (42, 1):
        m = rb.NETZE["a03-aniso"](2)
        m.load_state_dict(torch.load(rb.VALDO_MEMBERS[seed], map_location="cpu"), strict=True)
        models.append(m.to(rb.cv5.DEV).eval())
    t0 = time.time()
    for sid in ids:
        target = f"{pred}/{sid}_pred_proba.nii.gz"
        if os.path.exists(target):
            continue
        p = np.mean([rb.cv5.proba_fall(m, sid, nch=2)[0] for m in models], axis=0)
        back = np.ascontiguousarray(p.transpose(2, 1, 0))
        nib.save(nib.Nifti1Image(np.round(back * 255).astype(np.uint8), rb.cv5.FA["recs"][sid]["affine"]), target)
    json.dump(dict(ok=True, models=[rb.VALDO_MEMBERS[42], rb.VALDO_MEMBERS[1]], ensemble="mean", mod=a.mod,
                   grid=rb.GRID[a.mod], cases=ids, variant="N0", n="0", seed=0, strategy="STRAT"),
              open(f"{out}/pred_meta.json", "w"), indent=1)
    print(f"N0 DONE {a.mod}: {len(ids)} scans, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
