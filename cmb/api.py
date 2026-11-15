"""Python API: the ``cmb`` commands as functions, with the option names of the command line (without the dashes).

    from cmb import env, get_model, predict, preprocess, train, fine_tune, evaluate, cross_validate

    E = env()                          # the paths of .env (repository root)
    predict(E)                         # = cmb predict -i $CMB_SCAN -o $CMB_OUTPUT_DIR
    preprocess(E)                      # = cmb preprocess -i $CMB_INPUT_DIR -l $CMB_LABEL_DIR -b $CMB_MASK_DIR -o $CMB_GRID_DIR
    train(E, epochs=1)                 # = cmb train -i $CMB_GRID_DIR -m $CMB_MODEL_DIR (+ stage 2)
    fine_tune(E, epochs=1)             # = cmb fine_tune -i $CMB_GRID_DIR -m pre -o $CMB_FINETUNED_DIR

or with explicit arguments, e.g. ``predict("scan.nii.gz", o="out/")``, ``train("grid/", m="model/", ep=60)``.
Every function builds the argument list of the command line and runs it in this process.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional, Sequence

__all__ = ["get_model", "predict", "preprocess", "train", "fine_tune", "evaluate", "cross_validate"]

FLAGS = dict(o="-o", m="-m", l="-l", b="-b", ep="-ep", ilr="-ilr", bs="-bs", patches="--patches", frst="--frst",
             brain_mask="--brain-mask", device="--device", csf_rule="--csf-rule", no_bias_correction="--no-bias-correction")
SWITCHES = ("csf_rule", "no_bias_correction")          # store_true options of the command line


def _opts(**kw) -> List[str]:
    out: List[str] = []
    for k, v in kw.items():
        if v is None or (k in SWITCHES and not v):
            continue
        out += [FLAGS[k]] if k in SWITCHES else [FLAGS[k], str(v)]
    return out


def _cli(args) -> None:
    from cmb.cli import main
    main([str(a) for a in args])


def _env_int(E, name):
    v = getattr(E, name, "") if E is not None else ""
    return int(v) if str(v).strip() else None


def get_model(name: str = "valdo_t2") -> str:
    """Download (once, SHA-256 checked) and return the model folder."""
    from cmb.pretrained import get_model as g
    return g(name)


def predict(i, o: Optional[str] = None, m: Optional[str] = None, frst: Optional[str] = None,
            brain_mask: Optional[str] = None, device: str = "auto", csf_rule: bool = False,
            no_bias_correction: bool = False):
    """cmb predict. ``i`` = scan, folder, or ``env()`` (CMB_SCAN -> CMB_OUTPUT_DIR). Returns one dict per scan
    (name, n = CMB count, classes, output, csv); for a single scan the dict itself."""
    from cmb.cli import build_parser
    from cmb import predict_cli
    if isinstance(i, SimpleNamespace):
        i, o = i.CMB_SCAN, o or i.CMB_OUTPUT_DIR
    args = ["predict", "-i", i] + _opts(o=o, m=m, frst=frst, brain_mask=brain_mask, device=device, csf_rule=csf_rule,
                                        no_bias_correction=no_bias_correction)
    res = predict_cli.run(build_parser().parse_args([str(x) for x in args]))
    return res[0] if len(res) == 1 else res


def preprocess(i, o: Optional[str] = None, ir: str = "_T2starw", f: bool = False, b: Optional[str] = None,
               br: str = "_brainmask", l: Optional[str] = None, lr: str = "_cmb", fold: int = 5, cohort: str = "data") -> str:
    """cmb preprocess. ``i`` = raw folder, or ``env()`` (CMB_INPUT_DIR, CMB_LABEL_DIR, CMB_MASK_DIR, suffixes -> CMB_GRID_DIR;
    with brain masks given the scans count as skull-stripped, -f True). Returns the grid folder."""
    if isinstance(i, SimpleNamespace):
        E = i
        i, o, ir, b, br, l, lr = (E.CMB_INPUT_DIR, o or E.CMB_GRID_DIR, E.CMB_SCAN_SUFFIX, getattr(E, "CMB_MASK_DIR", None),
                                  getattr(E, "CMB_MASK_SUFFIX", br), getattr(E, "CMB_LABEL_DIR", None),
                                  getattr(E, "CMB_LABEL_SUFFIX", lr))
        f = f or bool(b)
    _cli(["preprocess", "-i", i, "-o", o, "-ir", ir, "-f", str(bool(f)), "-br", br, "-lr", lr, "-fold", fold,
          "--cohort", cohort] + _opts(b=b, l=l))
    return o


def train(i, m: Optional[str] = None, l: Optional[str] = None, ep: Optional[int] = None, epochs: Optional[int] = None,
          ilr: Optional[float] = None, bs: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
          patches: Optional[int] = None, frst: bool = True, device: str = "auto", stage2: str = "joint") -> str:
    """cmb train (stage 1 on all cases, the paper's recipe: 60 epochs, seeds 42 and 1) and, with stage2 = "joint" or
    "separate", ``cmb cross_validate --stage2`` into ``<m>/cv`` (stage 2 learns from out-of-fold candidates).
    ``i`` = grid folder or ``env()`` (CMB_GRID_DIR -> CMB_MODEL_DIR; CMB_PATCHES and CMB_SEEDS if set)."""
    E = i if isinstance(i, SimpleNamespace) else None
    if E is not None:
        i, m = E.CMB_GRID_DIR, m or E.CMB_MODEL_DIR
        patches = patches or _env_int(E, "CMB_PATCHES")
        seeds = seeds or ([int(s) for s in E.CMB_SEEDS.split()] if getattr(E, "CMB_SEEDS", "") else None)
    ep = epochs or ep or 60
    seeds = list(seeds or (42, 1))
    _cli(["train", "-i", i, "-m", m, "-ep", ep, "--seeds", *seeds, "--frst", str(bool(frst)), "--device", device]
         + _opts(l=l, ilr=ilr, bs=bs, patches=patches))
    if stage2 != "none":
        cross_validate(i, o=os.path.join(m, "cv"), l=l, ep=ep, patches=patches, stage2=stage2, device=device)
    return m


def fine_tune(i, o: Optional[str] = None, m: str = "pre", l: Optional[str] = None, ep: Optional[int] = None,
              epochs: Optional[int] = None, ilr: Optional[float] = None, ft_layers: str = "all",
              calibrate_only: bool = False, patches: Optional[int] = None, device: str = "auto") -> str:
    """cmb fine_tune: Phase 2 FT (ft_layers="all"), FTn ("norm_out") or the calibration-only arm; defaults 30 epochs,
    lr 1e-4. ``i`` = grid folder or ``env()`` (CMB_GRID_DIR -> CMB_FINETUNED_DIR, start = the VALDO T2* model)."""
    if isinstance(i, SimpleNamespace):
        E = i
        i, o = E.CMB_GRID_DIR, o or E.CMB_FINETUNED_DIR
        patches = patches or _env_int(E, "CMB_PATCHES")
    _cli(["fine_tune", "-i", i, "-m", m, "-o", o, "--ft_layers", ft_layers, "--calibrate_only", str(bool(calibrate_only)),
          "--device", device] + _opts(l=l, ep=epochs or ep, ilr=ilr, patches=patches))
    return o


def evaluate(i: str, o: str, m: str = "pre", l: Optional[str] = None, device: str = "auto") -> str:
    _cli(["evaluate", "-i", i, "-m", m, "-o", o, "--device", device] + _opts(l=l))
    return o


def cross_validate(i: str, o: str, l: Optional[str] = None, fold: int = 5, ep: Optional[int] = None,
                   patches: Optional[int] = None, stage2: str = "none", device: str = "auto") -> str:
    _cli(["cross_validate", "-i", i, "-o", o, "-fold", fold, "--stage2", stage2, "--device", device]
         + _opts(l=l, ep=ep, patches=patches))
    return o
