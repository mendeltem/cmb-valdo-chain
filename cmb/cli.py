"""``cmb``: the command line of the paper chain. Same command and option names as cmb_3d_unet_experiments (cmbx), which
follow MicrobleedNet: -i input, -l labels, -m model, -o output, -ep epochs, -ilr learning rate, -bs batch size.

    cmb predict -i scan.nii.gz [-o out/] [-m model/]       one scan or a folder (the public VALDO T2* model by default)
    cmb preprocess / train / fine_tune / evaluate / cross_validate      see ``cmb <command> --help``
    cmb get-model valdo_t2 | --list
    cmb selftest                                            no data, CPU, < 2 min
    cmb paths                                               where the code looks for data (environment variables)
"""
from __future__ import annotations

import argparse
import sys


def _common_train(p, defaults: str) -> None:
    p.add_argument("-ep", type=int, default=None, help=f"epochs ({defaults})")
    p.add_argument("-ilr", type=float, default=None, help="initial learning rate (see the table in the README)")
    p.add_argument("-bs", type=int, default=None, help="batch size (default: recipe, 4)")
    p.add_argument("--frst", default="True", help="FRST as the second input channel (True/False; paper: True)")
    p.add_argument("--patches", type=int, default=None, help="patches per epoch (recipe 800; small values for smoke tests only)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cmb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    from cmb import predict_cli as predict
    p = sub.add_parser("predict", help="microbleeds on one scan or a folder of scans",
                       description=predict.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    predict.add_arguments(p)

    p = sub.add_parser("preprocess", help="raw scans -> the paper's model grid (brain mask, FAST, inversion, grid, FRST)")
    p.add_argument("-i", required=True, help="folder with the raw scans")
    p.add_argument("-ir", required=True, help="scan name suffix at the END of the file name, e.g. _T2starw (<id>_T2starw.nii.gz); "
                   "searched recursively (flat folders and BIDS trees); names ending in -lr / -br are never scans")
    p.add_argument("-f", default="False", help="True: scans are already skull-stripped and bias-corrected (no FSL)")
    p.add_argument("-b", default=None, help="folder with brain masks (default: computed, see cmb predict --help)")
    p.add_argument("-br", default="_brainmask", help="brain mask suffix")
    p.add_argument("-l", default=None, help="folder with the CMB masks (training data)")
    p.add_argument("-lr", default="_cmb", help="CMB mask suffix")
    p.add_argument("-o", required=True, help="output grid folder (one <id>/ per case + manifest.json)")
    p.add_argument("-fold", type=int, default=5, help="number of cross-validation folds written into manifest.json")
    p.add_argument("--cohort", default="data", help="cohort name in the manifest (the joint/separate classifier groups by it)")
    p.add_argument("--profile", choices=["own", "valdo"], default="own",
                   help="own = the paper's pipeline for its own cohorts (cmb.transfer.build_private_grids: RAS+ reorientation, "
                        "brain = mask and image > 0; default); valdo = exactly the VALDO grid dev/gitter_d (no reorientation)")

    p = sub.add_parser("train", help="train the stage-1 U-Net on all cases (cmb.training.full_data)")
    p.add_argument("-i", required=True, help="grid folder from cmb preprocess")
    p.add_argument("-l", default=None, help="folder with <id>/<id>_label.nii.gz on the grid (default: -i)")
    p.add_argument("-m", required=True, help="model output folder (<seed>/modell.pt + meta.json)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1], help="one member per seed (paper: 42 1)")
    p.add_argument("--overwrite", action="store_true")
    _common_train(p, "paper: 60")

    p = sub.add_parser("fine_tune", help="fine-tune a model (Phase 2 FT / FTn) or re-choose its threshold only")
    p.add_argument("-i", required=True, help="grid folder of the local scans")
    p.add_argument("-l", default=None, help="labels (default: -i)")
    p.add_argument("-m", required=True, help="start model: 'pre' = the public VALDO T2* model, or a model folder")
    p.add_argument("-o", required=True, help="new model folder")
    p.add_argument("--ft_layers", choices=["all", "norm_out"], default="all",
                   help="all = full fine-tuning (FT); norm_out = only InstanceNorm affine + output head (FTn)")
    p.add_argument("--calibrate_only", default="False",
                   help="True: no training; threshold and minimum size re-chosen on the given scans (Phase 2 CAL arm)")
    _common_train(p, "paper: 30")

    p = sub.add_parser("evaluate", help="stage-1 maps and masks for a grid folder (+ lesion F1 if labels exist)")
    p.add_argument("-i", required=True, help="grid folder")
    p.add_argument("-m", default="pre", help="model folder (default: the public VALDO T2* model)")
    p.add_argument("-o", required=True, help="prediction folder")
    p.add_argument("-l", default=None, help="labels (default: -i)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    p = sub.add_parser("cross_validate", help="k-fold CV of stage 1 (code/turnier.py) and optionally stage 2")
    p.add_argument("-i", required=True, help="grid folder")
    p.add_argument("-l", default=None, help="labels (default: -i)")
    p.add_argument("-o", required=True, help="output folder")
    p.add_argument("-fold", type=int, default=5, help="5 (the paper's CV code needs 5 folds: test k, inner k+1 mod 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage2", choices=["joint", "separate", "none"], default="none",
                   help="stage-2 classifier on the out-of-fold candidates: joint = one for all cohorts (the paper's), "
                        "separate = one per cohort (also reports joint)")
    _common_train(p, "paper: 60")

    p = sub.add_parser("get-model", help="download a pretrained model (SHA-256 checked)")
    p.add_argument("name", nargs="?", default="valdo_t2")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dir", default=None, help="target folder (default: ~/.cache/cmb-valdo-chain/<name>)")

    sub.add_parser("selftest", help="no data, CPU: imports, grid, U-Net forward pass, metric")
    sub.add_parser("paths", help="print where the code looks for data")
    return ap


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    a = ap.parse_args(argv)
    from cmb import commands as C
    if a.cmd == "predict":
        from cmb import predict_cli as predict
        predict.run(a)
    elif a.cmd == "preprocess":
        C.preprocess(a)
    elif a.cmd == "train":
        a.ep = a.ep or 60
        C.train(a)
    elif a.cmd == "fine_tune":
        C.fine_tune(a)
    elif a.cmd == "evaluate":
        C.evaluate(a)
    elif a.cmd == "cross_validate":
        C.cross_validate(a)
    elif a.cmd == "get-model":
        from cmb import pretrained as PR
        if a.list:
            for n in PR.REGISTRY:
                print(PR.describe(n))
            return
        print(PR.get_model(a.name, a.dir))
        print(PR.describe(a.name))
    elif a.cmd == "selftest":
        from cmb import selftest
        selftest.main()
    elif a.cmd == "paths":
        from cmb import paths
        print(paths.describe())


if __name__ == "__main__":
    main()
