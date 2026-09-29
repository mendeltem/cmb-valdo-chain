"""Average the out-of-fold probability maps of several runs into a new "run" folder.

Why: a false alarm that only ONE network raises is halved by averaging, a microbleed that all networks see stays.
Runs that differ only in the random seed share the folds (dev/split.json or the manifest), so every case is
out-of-fold in all of them and the average is still an honest out-of-fold prediction. The result has the same layout
as a training run (fold<k>/<id>_pred_proba.nii.gz, 8 bit), so every other tool works on it unchanged:
cmb.transfer.evaluate, cmb.analysis.operating_point, cmb.stage2.candidates, ...

Next to the mean it writes <id>_pred_agree.nii.gz: the MINIMUM over the members -- "how sure is the least convinced
network?" -- as a ready-made agreement cue for a second stage.

    python -m cmb.analysis.ensemble --runs ergebnisse/turnier/a03-aniso-e60-gd ergebnisse/turnier/a03-aniso-s1-e60-gd \
        --out ergebnisse/turnier/ens2-a03-aniso-e60-gd
Private cohorts: give a private --out.
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

ROOT = "/home/uchralt/data/work/valdo-t2s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    runs, out = [absolute(r) for r in args.runs], absolute(args.out)

    index = []                                            # per run: case id -> (fold, path)
    for run in runs:
        found = {}
        for path in glob.glob(f"{run}/fold*/*_pred_proba.nii.gz"):
            found[os.path.basename(path)[:-len("_pred_proba.nii.gz")]] = (os.path.basename(os.path.dirname(path)), path)
        index.append(found)
    shared = sorted(set.intersection(*(set(f) for f in index)))
    written = 0
    for case in shared:
        folds = {f[case][0] for f in index}
        if len(folds) != 1:                               # not out-of-fold in the same fold everywhere: averaging would leak nothing,
            raise SystemExit(f"{case}: folds differ between runs ({folds}) -- refusing to mix")     # but the layout would lie
        first = nib.load(index[0][case][1])
        stack = np.stack([np.asarray(nib.load(f[case][1]).dataobj).astype(np.float32) for f in index])
        os.makedirs(f"{out}/{folds.copy().pop()}", exist_ok=True)
        for suffix, volume in (("proba", stack.mean(0)), ("agree", stack.min(0))):
            image = nib.Nifti1Image(np.round(volume).astype(np.uint8), first.affine, first.header)
            image.set_data_dtype(np.uint8)
            nib.save(image, f"{out}/{folds.copy().pop()}/{case}_pred_{suffix}.nii.gz")
        written += 1
    json.dump(dict(members=runs, cases=written), open(f"{out}/ensemble.json", "w"), indent=1)
    print(f"{written} cases averaged over {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
