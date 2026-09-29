"""Off-the-shelf baseline: SHIVA-CMB (Tsuchida et al., Sci Rep 2024).

Question answered here: how well does a freely available model, trained on 450 T2*-GRE and SWI
scans from six cohorts, detect microbleeds on OUR data without any adaptation?

Licence of the model weights: CC BY-NC-SA 4.0 -- research use only, not for clinical purposes;
anything derived from the weights (e.g. a fine-tuned model) must carry the same licence.
Code https://github.com/pboutinaud/SHIVA_CMB (commit 4367d93), weights "v2", three folds.

What the model expects (README): 1 mm isotropic voxels, volume 160 x 214 x 176, intensities
min-max scaled to [0, 1] with the maximum set to the 99th percentile of the brain voxels,
brain roughly centred. Orientation is not documented; we use RAS+ (nibabel canonical), which is
what the authors' own pipeline (SHiVAi) conforms to. NOTE: the network sees the image as acquired
(microbleeds dark) -- not our inverted image.

Pipeline per case
-----------------
1. load the bias-corrected T2* and the brain mask, reorient both to RAS+
2. resample to 1 mm (image linear, mask nearest), crop / pad to 160 x 214 x 176 around the
   centre of mass of the mask, normalise
3. average the predictions of the three fold models
4. resample the probability map back onto the evaluation grid (linear), threshold at 0.5
5. score with the project metric (26-connectivity, touch rule) and the official VALDO rule

Run inside the dedicated environment (TensorFlow lives there, not in ``dl``):

    CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/shiva/bin/python -m cmb.baselines.shiva_cmb \
        --cohort valdo --out ergebnisse/baselines/shiva-cmb-valdo
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

ROOT = "/home/uchralt/data/work/valdo-t2s"
SHIVA_SHAPE = (160, 214, 176)
MODEL_GLOB = "/home/uchralt/software/SHIVA_CMB/modelle/SWI-CMB/*.tf_inference"
THRESHOLD = 0.5


# --------------------------------------------------------------------------- case lists
def valdo_cases() -> List[Dict[str, str]]:
    """The 57 cross-validation cases (the 15 held-out cases stay untouched)."""
    held_out = set(json.load(open(f"{ROOT}/dev/split.json"))["validation"])
    cases = []
    for folder in sorted(glob.glob(f"{ROOT}/dev/preproc/sub-*")):
        case_id = os.path.basename(folder)
        if case_id in held_out:
            continue
        cases.append(dict(
            id=case_id,
            image=f"{folder}/fast_restore.nii.gz",                       # bias-corrected, not inverted
            brain=f"{ROOT}/dev/preproc_roh/{case_id}/{case_id}_image.nii.gz",   # > 0 inside the mask
            eval_label=f"{ROOT}/dev/gitter_d/{case_id}/{case_id}_label.nii.gz",  # evaluation grid
        ))
    return cases


def catalina_cases(modality: str) -> List[Dict[str, str]]:
    """Private cohort (never public, sums only). SHIVA-CMB has never seen these scans, so this --
    unlike VALDO, which is part of SHIVA's training data -- is a fair test of the off-the-shelf model.
    Evaluation happens on the native image grid, where the manual masks live."""
    listing = json.load(open("/home/uchralt/data/work/mb-arena/faelle.json"))[modality]["faelle"]
    cases = []
    for entry in listing:
        corrected = glob.glob(os.path.dirname(entry["bild"]).replace(
            f"/{entry['id']}/anat", f"/derivatives/biascorr/{entry['id']}/anat") + "/*biascorr*.nii.gz")
        cases.append(dict(id=entry["id"], image=corrected[0] if corrected else entry["bild"],
                          brain=entry["hirn"], eval_label=entry["maske"]))
    return cases


# --------------------------------------------------------------------------- preprocessing
def conform(image: nib.Nifti1Image, brain: nib.Nifti1Image) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Return the network input ``(160, 214, 176)`` and a NIfTI describing that grid in world space."""
    image_1mm = resample_to_output(nib.as_closest_canonical(image), voxel_sizes=1.0, order=1)
    brain_1mm = resample_from_to(nib.as_closest_canonical(brain), image_1mm, order=0)
    data = np.asarray(image_1mm.dataobj, dtype=np.float32)
    mask = np.asarray(brain_1mm.dataobj) > 0

    centre = np.round(np.array(np.nonzero(mask)).mean(axis=1)).astype(int)
    start = centre - np.array(SHIVA_SHAPE) // 2                           # may be negative -> padding
    out = np.zeros(SHIVA_SHAPE, np.float32)
    out_mask = np.zeros(SHIVA_SHAPE, bool)
    src = tuple(slice(max(s, 0), min(s + n, d)) for s, n, d in zip(start, SHIVA_SHAPE, data.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    out[dst] = data[src]
    out_mask[dst] = mask[src]

    out *= out_mask
    top = np.percentile(out[out_mask], 99)
    out = np.clip(out / max(float(top), 1e-6), 0.0, 1.0)

    # World coordinates of the cropped grid: shift the 1 mm affine by the crop start.
    affine = image_1mm.affine.copy()
    affine[:3, 3] = affine[:3, :3] @ start + affine[:3, 3]
    return out, nib.Nifti1Image(out_mask.astype(np.uint8), affine)


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=["valdo", "catalina-t2star", "catalina-swi"], default="valdo")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke test)")
    args = parser.parse_args()

    import tensorflow as tf                       # imported late: slow, and only needed here
    sys.path.insert(0, f"{ROOT}/code")
    from metric import hits, summarize
    from metric_valdo import case_valdo, summarize_valdo

    out_dir = args.out if os.path.isabs(args.out) else f"{ROOT}/{args.out}"
    os.makedirs(out_dir, exist_ok=True)
    model_paths = sorted(glob.glob(MODEL_GLOB))
    assert len(model_paths) == 3, f"expected 3 fold models, found {model_paths}"
    models = [tf.saved_model.load(p) for p in model_paths]

    cases = valdo_cases() if args.cohort == "valdo" else catalina_cases(args.cohort.split("-")[1])
    if args.limit:
        cases = cases[:args.limit]
    rows, rows_valdo = [], []
    for number, case in enumerate(cases, 1):
        t0 = time.time()
        network_input, grid = conform(nib.load(case["image"]), nib.load(case["brain"]))
        batch = network_input[np.newaxis, ..., np.newaxis]
        probability = np.mean([np.asarray(m.serve(batch))[0, ..., 0] for m in models], axis=0)
        probability *= np.asarray(grid.dataobj) > 0

        label_img = nib.load(case["eval_label"])
        on_eval_grid = resample_from_to(nib.Nifti1Image(probability.astype(np.float32), grid.affine),
                                        label_img, order=1)
        prediction = np.asarray(on_eval_grid.dataobj) > THRESHOLD
        reference = np.asarray(label_img.dataobj) > 0
        if reference.ndim == 4:                      # a few private masks carry a singleton 4th axis
            reference, prediction = reference[..., 0], prediction.reshape(reference.shape[:3])
        nib.save(nib.Nifti1Image(prediction.astype(np.uint8), label_img.affine),
                 f"{out_dir}/{case['id']}_pred.nii.gz")

        rows.append(hits(reference, prediction))
        rows_valdo.append(case_valdo(reference, prediction.astype(np.uint8), label_img.affine))
        print(f"[{number}/{len(cases)}] {case['id']}: reference {rows[-1]['n_ref']}, predicted "
              f"{rows[-1]['n_pred']}, TP {rows[-1]['tp']}, FP {rows[-1]['fp']}  ({time.time() - t0:.0f} s)", flush=True)

    ours, valdo = summarize(rows), summarize_valdo(rows_valdo)
    summary = dict(model="SHIVA-CMB v2 (3 folds), no adaptation", n_cases=len(rows), threshold=THRESHOLD,
                   project_metric={k: round(float(ours[k]), 4) for k in ("f1", "precision", "sensitivity", "fp_per_case")},
                   valdo_metric={k: round(float(valdo[k]), 4) for k in ("f1_case_median", "f1_case_q1", "f1_case_q3", "f1_pooled")})
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
