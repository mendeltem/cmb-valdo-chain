"""Re-predict the out-of-fold cases of a finished run with flip test-time augmentation (TTA).

Why: training already flips every patch along z, y and x at random, so the network has seen all eight mirror images.
Averaging its answer over these eight views at test time is a free ensemble -- no extra training -- and is expected to
act like a small seed ensemble: responses that depend on one orientation (often false alarms) are damped, stable
responses stay. In the WMH line of this lab the same trick gave +0.8 to +1.4 Dice points.

How: for fold k the stored network fold<k>/modell.pt predicts exactly the cases it predicted before (the ones with a
probability map in fold<k>/), so the result is still out-of-fold. The output has the layout of a training run
(fold<k>/<id>_pred_proba.nii.gz, probability * 255), so evaluate / operating_point / ensemble / stage2 work unchanged.
The sliding window itself is the project's legacy inference (cv5.proba_fall); only the network is wrapped.

    python -m cmb.analysis.tta_predict --run ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --grid dev/gitter_d \
        --out ergebnisse/turnier/a03-aniso-e60-gd-tta
    python -m cmb.analysis.tta_predict --run /private/run --net a03-aniso --manifest /private/manifest.json --out /private/run-tta

Needs the GPU (8 x the normal inference, about half a minute per case) -- queue it, do not start it next to a training.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")

SPATIAL_AXES = (2, 3, 4)                                     # (batch, channel, z, y, x)
FLIPS = [dims for n in range(4) for dims in itertools.combinations(SPATIAL_AXES, n)]     # 8 views, () first


class FlipAverage(torch.nn.Module):
    """Mean PROBABILITY over the eight mirror views, returned as a logit because the caller applies the sigmoid."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        for dims in FLIPS:
            logits = self.net(torch.flip(x, dims) if dims else x)
            probability = torch.sigmoid(torch.flip(logits, dims) if dims else logits).float()
            total = probability if total is None else total + probability
        mean = (total / len(FLIPS)).clamp(1e-6, 1 - 1e-6)
        return torch.log(mean / (1 - mean))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="finished run: fold<k>/modell.pt and fold<k>/<id>_pred_proba.nii.gz")
    parser.add_argument("--net", required=True, help="architecture name as used in code/netze.py")
    parser.add_argument("--grid", help="VALDO grid folder, e.g. dev/gitter_d")
    parser.add_argument("--manifest", nargs="+", help="alternative to --grid (other cohorts)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--channels", type=int, default=2, help="2 = T2* + FRST, 1 = T2* only")
    args = parser.parse_args()
    if bool(args.grid) == bool(args.manifest):
        raise SystemExit("give exactly one of --grid and --manifest")
    import cv5                                   # legacy inference code (sliding window with half stride)
    import turnier                               # noqa: F401  (registers the tournament networks in cv5)
    from netze import NETZE

    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    run, out = absolute(args.run), absolute(args.out)
    if args.manifest:
        turnier.mit_manifest([absolute(m) for m in args.manifest])()
    else:
        cv5.GITTER = absolute(args.grid)
        cv5.lade_daten()
    records = cv5.FA["recs"]

    done = 0
    for fold_dir in sorted(glob.glob(f"{run}/fold[0-9]")):
        fold = os.path.basename(fold_dir)
        case_ids = sorted(os.path.basename(p)[:-len("_pred_proba.nii.gz")] for p in glob.glob(f"{fold_dir}/*_pred_proba.nii.gz"))
        net = NETZE[args.net](args.channels)
        net.load_state_dict(torch.load(f"{fold_dir}/modell.pt", map_location="cpu"))
        model = FlipAverage(net.to(cv5.DEV)).eval()
        os.makedirs(f"{out}/{fold}", exist_ok=True)
        for case_id in case_ids:
            probability, _ = cv5.proba_fall(model, case_id, nch=args.channels)                       # (z, y, x)
            back = np.ascontiguousarray(probability.transpose(2, 1, 0))                            # -> (x, y, z)
            nib.save(nib.Nifti1Image(np.round(back * 255).astype(np.uint8), records[case_id]["affine"]),
                     f"{out}/{fold}/{case_id}_pred_proba.nii.gz")
            done += 1
            print(f"[{done}] {fold} {case_id}  max probability {probability.max():.2f}", flush=True)
    json.dump(dict(source_run=run, net=args.net, views=len(FLIPS), cases=done), open(f"{out}/tta.json", "w"), indent=1)
    print(f"{done} cases re-predicted with {len(FLIPS)} mirror views -> {out}")


if __name__ == "__main__":
    main()
