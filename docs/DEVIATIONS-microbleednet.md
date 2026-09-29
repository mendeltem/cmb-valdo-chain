# Deviations from the original MicrobleedNet (Sundaresan et al.)

**Purpose (user task 2026-09-19):** be able to say at any time WHAT is different in ours compared to the original and HOW MUCH.
This file is updated on EVERY change to preprocessing, network, training, post-processing or metric
(new line or changed line with date). Strength: **none** (identical) / **small** (repair, does not change the
procedure) / **medium** (same idea, different parameters) / **large** (different procedure).

## Reference state of the original

- Original: https://github.com/v-sundaresan/microbleed-detection , state **commit 958f1cb from 2026-03-19**
  ("Merge pull request #7 from g-shanker/main"). Checked online on 2026-09-19: that is the newest commit there.
- Local copy: `~/software/microbleed-detection` = fork `github.com/mendeltem/custom_microbleednet`,
  = original 958f1cb + 1 fork commit + uncommitted changes in 3 files (layer A below).
- Read from the source code: `microbleednet/scripts/data_preparation.py` (preprocessing). Everything below marked
  "(note)" comes from my project notes of 11 Sep and has NOT been checked against the original today.

## Layer A -- our fork `custom_microbleednet` against the original (everything here is OUR change)

User 19 Sep: the changes in the fork belong to our contribution and are therefore part of the comparison. Fully
broken down with `git diff 958f1cb 0aa3acf` (fork commit from 2026-07-13) and `git diff` (uncommitted).
"Read from the source code" means: call and signature compared, NOT executed.

| No | File | Original (958f1cb) | Fork | Strength | Why / evidence | State |
|---|---|---|---|---|---|---|
| A1 | `commands.py`, call to the CDisc fine-tuning | `main(..., aug=..., save_cp=..., save_wei=..., model_dir=...)` | `perform_augmentation=, save_checkpoint=, save_weights=, ...` | small (repair) | `main()` does not know the names `aug`/`save_cp`/`save_wei` (signature: `perform_augmentation, save_checkpoint, save_weights, ...`) -> a TypeError is to be expected in the original, the CDisc fine-tuning does not start via the command line. Read from the source code. | committed 0aa3acf |
| A2 | `cdet_finetune_function.py`, datasets | `CDetPatchDataset(pos, neg, perform_augmentations=True)` | `CDetPatchDataset(pos, neg, ratio='1:1', ...)` | small (repair) + **medium (fixed setting)** | `__init__(minority, majority, ratio, ...)` requires `ratio` -> original call without `ratio` = TypeError. With the repair, the positive:negative ratio is fixed at **1:1** ('2:1', 'random' would also be possible). Read from the source code. | committed |
| A3 | `cdisc_finetune_function.py`, datasets | `CDiscPatchDataset(NEGATIVE, POSITIVE, ...)`, without `ratio` | `CDiscPatchDataset(POSITIVE, NEGATIVE, ratio='1:1', ...)` | **medium** | Signature is `(minority_class, majority_class, ratio)`: in the original the classes are passed swapped and `ratio` is missing. Read from the source code. | committed |
| A4 | `cdisc_finetune_function.py`, freezing | `freeze_layers_for_finetuning` only knows the U-Net layer names; the classifier head (classconvfirst, classdown1/2, fc1-3) ALWAYS stays frozen | head is explicitly unfrozen; additionally `inpconv` uncommitted | small (bug fix), effect large | Without this, CDisc collapses during fine-tuning to "all negative" (accuracy 0.5, 0 positives) -- observed in the MB arena. | head committed, `inpconv` not |
| A5 | `utils.py` | `load_model` present | second definition `load_model(checkpoint_path, model, mode)` appended at the end of the file | small | overwrites the first; whether the behaviour differs has NOT yet been compared (open). | committed |
| A6 | `cdet_evaluate_function.py` | -- | two `print(model path)` | none | diagnostic output | committed |
| A7 | `cdet_evaluate_function.py`, candidate threshold | `predictions > 0.2` | `> 0.1` | medium | Pretrained weights reach at most 0.217 per case on catalina-SWI; at 0.2 there are NO candidates in 59/61 cases, at 0.1 in 54/61 (mean 2.75 per case). Measured 7 Sep (diagnose.json). | NOT committed |
| A8 | `data_preparation.py`, brain mask (3 places) | `brain_mask = (image > 0)` | `binary_fill_holes(image > 0)` | small | voxels with value 0 INSIDE the brain (very dark CMB/vessels) fell out of the mask and were deleted during masking. | NOT committed |
| A9 | repo hygiene | build artifacts, egg-info, caches, `scripts-3.10/` in the repo | removed, `.gitignore` | none | 9645 lines deleted, no behaviour affected | committed |

**Classification for the thesis:** A1-A3 mean that the published fine-tuning path of the original (state 958f1cb)
does not run without intervention according to the source code; A4 means that -- once runnable -- it does not
train the classifier. This is an independent finding about the original, unrelated to our VALDO networks. Before a statement
to the outside: A1-A3 must be RUN ONCE (fresh clone of 958f1cb, call as in the example scripts) and the error message
recorded verbatim.

The function `inpaint_vessels` (vessel inpainting) is **unchanged** in the local copy compared to 958f1cb --
`frangi(sigmas=(0.5,1.2,0.2), alpha=0.9, beta=20)`, structure-tensor linearity, KMeans k=2 per slice, rescue at
eccentricity < 0.9 and solidity > 0.5 thus come from the original.

## Layer B -- our VALDO procedure against the MicrobleedNet procedure

| Component | Original MicrobleedNet | Ours (state 2026-09-19) | Strength | Why / measured | File |
|---|---|---|---|---|---|
| Bias correction | not part of the Python chain (FSL preprocessing via shell script, note) | FSL FAST -B before the chain, 72/72 | small | SD in the brain, cohort 1 about -12%, cohort 2 almost 0 | `code/preprocess.py` |
| Brain mask | image > 0 | supplied VALDO mask (image > 0, holes filled). SynthSeg brain + 2 voxels (`--maske synthseg`) checked on 19 Sep: F1 0.421 vs 0.470, NOT adopted | small | VALDO mask is 40% non-brain in cohort 3 (2533 ml), cohort 2 16% | `code/turnier.py` |
| Vessel inpainting | `inpaint_vessels` (see above) | so far adopted identically (gitter_c). **Under review:** none (gitter_d) or inpainting 2 (gitter_e/f/g): beta 0.5, scales in mm, one threshold per volume, only thin components >= L mm long | so far none -> large | original hits 109/236 VALDO CMB, 17 invisible afterwards, inpaints 12-17% of the brain; inpainting 2 (3%, 8 mm): 1/236, 0.9-1.5%. **Training without inpainting: F1 0.591 vs 0.455 with original inpainting, paired +0.136 [+0.065; +0.226] (19 Sep, 4/5 folds) -> original inpainting dropped**; inpainting 2 vs "none" running | `code/uebermalung2.py` |
| Inversion / scale | `1 - image/maximum`, masked | identical. z-normalisation per case (`--norm z`) checked on 19 Sep in a package with augmentation: F1 0.410 vs 0.470; evaluated individually on 28 Sep: VALDO -0.005, zero-shot T2* -0.004 / SWI -0.022, all n.s. Landmark alignment (`--norm landmarken`, Nyul-Udupa, reference `dev/landmarken_valdo.json`): VALDO -0.064, zero-shot T2* -0.048 / SWI -0.034 (both confirmed), mixed VALDO share 0.355 vs 0.582. Neither adopted (WORKLOG.md 5ac) | none | scale not harmonised between cohorts (5th-95th percentile 0.25-0.74 vs 0.65-0.98) | `code/turnier.py` |
| FRST | code (read 19 Sep): `fast_radial_symmetry_xfm` per slice, `radii = [2, 3]`, alpha 2, on the original grid, min-max normalised. PAPER 2023: mean over four radii 2/3/4/6 -- code and paper diverge | own FRST (`code/frst.py`, Loy & Zelinsky, continuous), radii [2,3,4,6] voxels = 1-3 mm on the common grid, 99.9th-percentile normalisation | medium | top-1% hit rate at the CMB 82% -> 89% (11 Sep); without inpainting 96% (19 Sep) | `code/frst.py`, `code/gitter_c_bauen.py` |
| Grid | original resolution per case | all cases at 0.5 x 0.5 x 1.0 mm, cropped to the brain (grid C) | large | three cohorts with 0.45x0.45x4 / 0.49x0.49x0.8 / 1x1x3-4 mm in ONE network | `code/gitter_c_bauen.py` |
| Network | two-stage: CDet (candidates, 48^3) + CDisc (classifier, 24^3), teacher-student distillation (note) | single-stage: anisotropic 3D U-Net `a03-aniso` (first stage only in the layer), 22.5 million parameters; tournament over 10 architectures | large | a03 F1 0.404 vs Attention U-Net 0.245, paired +0.159 [+0.091; +0.234]; two-stage version built on 19 Sep and tested nested (`cmb/stage2/`): 0.599-0.616 vs 0.627 without second stage -- not adopted (too few candidates) | `code/netze.py` |
| Input channels | image + FRST | identical (T2* + FRST). Control without FRST (20 Sep): F1 0.535 vs 0.588, paired -0.053 [-0.116; +0.004] -> FRST stays | none | FRST helps as a pointer, but does not separate candidates (AUC 0.56) | `code/cv5.py` |
| Patch / sampling | 48^3 non-overlapping, per class (note) | 38x62x94 voxels (38x31x47 mm), 800 per epoch, 60% with CMB (random position in the patch) / 25% brain / 15% arbitrary; 1:2 collapses (5g); Sundaresan-like 24x24x24 mm checked on 22 Sep: F1 0.369 vs 0.585, paired -0.216 [-0.277; -0.156], NOT adopted; 48-mm cube: 0.569, -0.016 n.s., not adopted -- the recipe is at the optimum (WORKLOG.md 5t) | large | centred CMB -> network learned "lesion = centre" (11 Sep) | `code/cv5.py` |
| Loss | weighted cross-entropy + Dice with pixel weights (note) | BCE + Dice, unweighted | medium | only recipe that measurably learned on 11 Sep | `code/cv5.py` |
| Training duration | -- (not checked) | until 19 Sep: <= 40 epochs with early stopping; now 60 fixed epochs; 300 queued | -- | 40 -> 60 fixed: F1 0.404 -> 0.470 raw; early stopping left one fold at epoch 5 | `code/turnier.py --epochen` |
| Augmentation | own module `augmentations` (note) | mirroring only. Rotation/scale/contrast/noise (`--aug`) checked on 19 Sep, not adopted | medium | package with z-normalisation: -0.060 [-0.135; +0.011], all 5 folds lower | `code/turnier.py` |
| Post-processing | distance to brain edge >= 8 voxels, shape filter per slice (eccentricity > 0.4 and solidity < 0.5 -> removed), < 2 voxels removed (note) | threshold + minimum size over the inner fold, blob guard (> 2100 voxels removed), **CSF rule** (majority label SynthSeg CSF -> removed) | large | CSF rule +0.05 to +0.09 F1 without TP loss; an edge distance like in the original was never measured for us (open) | `code/cv5.py`, `code/gewebekarte.py` |
| Teacher-student | CDisc is trained via knowledge distillation from a teacher network (note; soft labels instead of hard 0/1) | **NOT checked** (state 2026-09-22). Our shared 3D CNN learns directly from hard labels. Open gap, asked about by the user on 22 Sep; the probability map of the first stage as a soft target would be an obvious choice -- fits the data scarcity on VALDO (278 candidates) | open | -- | -- |
| Metric | -- | 26-connectivity components, touch rule (`metric.py`); additionally the official VALDO rule (`metric_valdo.py`) | -- | both are reported | `code/metric.py` |
| Evaluation | -- | 5-fold CV with inner fold, 15 cases held out | -- | -- | `dev/split.json` |

## Open / still to check against the original
- Loss, patch strategy, augmentation, post-processing and training duration of the original are listed here from my
  notes, not from today's source code -- check against the code the next time it is touched and remove "(note)".
- RUN A1-A3 on a fresh clone of the original and record the error messages verbatim; compare A5 (duplicate `load_model`).
- A7, A8 and the `inpconv` part of A4 are not committed; without a commit they are lost on a fresh clone.

## Change log of this file
- 2026-09-19 created (Claude). Reference state 958f1cb confirmed online.
- 2026-09-19 layer A fully broken down (user: the fork changes are our contribution): A1-A9.
- 2026-09-19 z-normalisation + augmentation checked and discarded (rows inversion/scale, augmentation).
- 2026-09-19 FRST row: radii read from the code, deviation of code vs paper noted. Comparison Sundaresan/VALDO: research/sundaresan-vs-valdo.md.
- 2026-09-19 15:30 original inpainting removed (+0.136 confirmed without it); narrow input mask checked and discarded.
- 2026-09-19 evening: second stage checked and discarded (row network); post-processing: shared threshold 0.3 + minimum size 2 mm3, cross-wise 0.633.
- 2026-09-20 02:40: FRST confirmed (without: -0.053), grid z = 1 mm confirmed (z = 2 mm: -0.036); bias correction interim state -0.058 without.
- 2026-09-20 14:35: loss -- original cross-entropy (+ distillation), ours BCE+Dice; NEW option `--verlust blob` (one term per bleed, Kofler 2023), run queued, no result yet (strength: open). Location analysis (`cmb/analysis/fp_location.py`): original has no location filter, we only have the CSF rule; further location rules measurably bring nothing (WORKLOG.md 5i).
- 2026-09-20 evening: loss/output -- lesion-wise loss measured (VALDO: sensitivity of small bleeds 0.52 -> 0.64, F1 -0.044 n.s., strength: open, catalina running); NEW centre map as auxiliary output (`a12-anisozentrum`, `--verlust zentrum`, after CenSynCMB 2026), run queued. Second stage: original two-stage with distillation; for us also no gain on 961 catalina candidates against the pure threshold (WORKLOG.md 5j). Inference: mirror TTA and seed ensemble as tools (`cmb/analysis/tta_predict.py`, `ensemble.py`); the original has neither.
- 2026-09-20 late: training -- NEW options `--verlust fnrw` (reweighting by misses, CenSynCMB) and `--harte-negative` (online mining of own false positives; the original does not mine, its second stage learns on the candidates of the first); runs queued, strength: open. Inference: mirror TTA measured -- VALDO +0.02, catalina +-0 (strength: weak).
- 2026-09-20 22:55: centre map as auxiliary output measured -- F1 -0.054 [-0.110; -0.006] at the fixed cell, sensitivity of small bleeds 0.52 -> 0.66, false positives 86 -> 143; more sensitive like the lesion-wise loss, but not better (strength: negative; not adopted). Hard negatives run before fnrw.
- 2026-09-21 early: hard negatives (+0.000) and reweighting by misses (-0.030 n.s.) measured, not adopted; blob second seed -0.063, blob on SWI level. NEW option: tissue map (SynthSeg) as five input channels (`--gewebe`); the original has no anatomy input. Runs queued, strength: open.
- 2026-09-21 07:10: sampling/loss of the original checked AT THE SOURCE (user question "why does 1:x work for Sundaresan?"): in the code (commit 958f1cb, `cdet_train_function.py` line 80, `cdisc_train_function.py` line 123, `datasets.py`) both stages are set to `ratio='1:1'` (one patch with CMB per patch without); the class otherwise only knows '2:1' (TWO positives per negative) and 'random' -- there are no more negatives than positives in the original. The paper (Frontiers Neuroinf. 2023) states no ratio; it absorbs the imbalance in the LOSS: CE + Dice with 10x weighting of the CMB voxels; patches 48^3 (detection) resp. 24^3 (discriminator), batch 8, 100 epochs with early stopping (patience 20), augmentation x10 / x5. For us: 60% patches with CMB (1.5:1), no voxel weighting; 1:2 collapses (WORKLOG.md 5g). This verifies the table row "Patch / sampling".
- 2026-09-22 07:43: row patch/sampling -- Sundaresan-like 24-mm patches measured and discarded (-0.216 confirmed; WORKLOG.md 5t).
- 2026-09-22 10:17: 48-mm patches measured: -0.016 n.s., not adopted; dose series 24 / recipe / 48 mm completed (WORKLOG.md 5t).
- 2026-09-22 22:26: row teacher-student added (never checked, user question); QC page with 3 columns for catalina.
- 2026-09-28 landmark and z-normalisation evaluated individually (runs 23/24 Sep): neither adopted; the original's scale stays (row inversion/scale, WORKLOG.md 5ac).
