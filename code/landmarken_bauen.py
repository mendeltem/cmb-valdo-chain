"""Creates the reference landmarks for `turnier.py --norm landmarken`.

Why separate and as a file: the alignment needs a FIXED target that is identical for training and prediction --
even for a foreign cohort added later. A file can be shipped and checked; a reference computed
in code would silently depend on the cases just loaded.

    python code/landmarken_bauen.py --gitter dev/gitter_d --aus dev/landmarken_valdo.json

WHAT MUST BE DISCLOSED HERE: the reference is the median of the per-case percentiles over the given cases. If
one uses the 57 cross-validation cases for this, the intensity distribution of EVERY case flows with weight 1/57 into the scale
with which it is later also predicted. These are pure intensity percentiles, not labels -- the usual approach
in the literature and a very small influence, but an influence. Anyone who wants to rule this out builds the reference per
fold from its training cases (`--faelle` with the respective list) and passes it with `--landmarken`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/code")
from turnier import LANDMARKEN_PERZENTILE, landmarken_bestimmen     # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gitter", required=True, help="Folder with <id>/<id>_image.nii.gz")
    p.add_argument("--faelle", nargs="*", help="only these IDs (default: all in the grid)")
    p.add_argument("--aus", required=True)
    a = p.parse_args()
    gitter = a.gitter if os.path.isabs(a.gitter) else f"{ROOT}/{a.gitter}"
    ids = a.faelle or sorted(os.path.basename(d) for d in glob.glob(f"{gitter}/*") if os.path.isdir(d))
    bilder = []
    for i in ids:
        pfad = f"{gitter}/{i}/{i}_image.nii.gz"
        if os.path.exists(pfad):
            bilder.append(np.asarray(nib.load(pfad).dataobj).astype(np.float32))
    if not bilder:
        raise SystemExit(f"no images in {gitter}")
    werte = landmarken_bestimmen(bilder)
    aus = a.aus if os.path.isabs(a.aus) else f"{ROOT}/{a.aus}"
    json.dump(dict(perzentile=list(LANDMARKEN_PERZENTILE), landmarken=[round(float(v), 6) for v in werte],
                   gitter=gitter, n_faelle=len(bilder), faelle=sorted(ids) if a.faelle else "alle"),
              open(aus, "w"), indent=1)
    print(f"{len(bilder)} cases -> landmarks " + "  ".join(f"P{q}={v:.4f}" for q, v in zip(LANDMARKEN_PERZENTILE, werte)))
    print("written to", aus)


if __name__ == "__main__":
    main()
