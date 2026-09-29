"""Zero-shot transfer: apply the VALDO-trained networks to another cohort without any training.

The five fold models of one run are averaged (that is what one would deploy) and the probability
maps are written in the same layout as a training run, so ``cmb.transfer.evaluate`` can score them:

    <out>/fold<k>/<id>_pred_proba.nii.gz          (uint8, probability * 255; k = the case's own fold)

    python -m cmb.transfer.zero_shot --models ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso \
        --manifest <other-cohort>/gitter/manifest.json --out <other-cohort>/transfer/zero-shot
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True, help="run folder containing fold0..fold4/modell.pt")
    parser.add_argument("--net", required=True, help="architecture name as used in code/netze.py")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--landmarks", default="dev/landmarken_valdo.json",
                        help="Referenzdatei fuer --norm landmarken; MUSS dieselbe sein wie beim Training")
    parser.add_argument("--norm", choices=["max", "z", "p995", "landmarken"], default="max",
                        help="Intensitaetsskala; MUSS zu der des Trainings passen. max = durch das Maximum im Hirn (Vorgabe, wie im Original). "
                             "landmarken ist die gemessen beste Angleichung (Kontrast-Spanne 0.018 gegen 0.060 heute), z die schlechteste "
                             "von sechs (0.362) -- ARBEIT.md 5aa.")
    args = parser.parse_args()
    import cv5                                   # legacy inference code (sliding window with half stride)
    import turnier
    from netze import NETZE

    models_dir = args.models if os.path.isabs(args.models) else f"{ROOT}/{args.models}"
    lader = turnier.mit_manifest(args.manifest)   # fills cv5.FA with the cases of the manifest
    if args.norm == "z":                          # the intensity scale MUST match the one the network was trained with
        lader = turnier.mit_z_normierung(lader)
    elif args.norm == "p995":
        lader = turnier.mit_perzentil_normierung(lader)
    elif args.norm == "landmarken":
        lader = turnier.mit_landmarken_normierung(
            lader, args.landmarks if os.path.isabs(args.landmarks) else f"{ROOT}/{args.landmarks}")
    lader()
    cases = [e for p in args.manifest for e in json.load(open(p))]
    nets = []
    for k in range(5):
        net = NETZE[args.net](2)
        net.load_state_dict(torch.load(f"{models_dir}/fold{k}/modell.pt", map_location="cpu"))
        nets.append(net.to(cv5.DEV).eval())

    for number, case in enumerate(cases, 1):
        record = cv5.FA["recs"][case["id"]]
        mean = np.mean([cv5.proba_fall(net, case["id"], nch=2)[0] for net in nets], axis=0)      # (z, y, x)
        folder = f"{args.out}/fold{case['fold']}"; os.makedirs(folder, exist_ok=True)
        back = np.ascontiguousarray(mean.transpose(2, 1, 0))                                     # -> (x, y, z)
        nib.save(nib.Nifti1Image(np.round(back * 255).astype(np.uint8), record["affine"]),
                 f"{folder}/{case['id']}_pred_proba.nii.gz")
        print(f"[{number}/{len(cases)}] {case['id']}  max probability {mean.max():.2f}", flush=True)
    json.dump(dict(models=models_dir, net=args.net, manifests=args.manifest, ensemble_of=5),
              open(f"{args.out}/zero_shot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
