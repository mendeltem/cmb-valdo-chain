# Cerebral microbleeds on T2*: the original under the microscope and a well-founded improvement

Working version in the style of a master's thesis (user request 2026-09-19). Rule for every intervention:
**Original -> Observation (measured) -> Hypothesis -> Change -> Experiment -> Result -> Decision -> Limits.**
No change without measurement before and after; "measured" and "interpreted" are kept separate; negative results
stay in. Numbers come from `protokoll.md` (there with date and file), the deviation table from
`DEVIATIONS-microbleednet.md`. Private cohort data (catalina) appear only as sums.

## 1 Research questions
1. Where does the published MicrobleedNet procedure (Sundaresan et al.; code commit 958f1cb) lose performance on
   heterogeneous T2*-GRE data (VALDO Task 2: three cohorts, 0.8 to 4 mm slice thickness) -- in preprocessing, in the
   network, in training, or in post-processing?
2. Which changes demonstrably fix this, measured by lesion F1 in stratified 5-fold cross-validation?
3. Does the gain hold on an independent cohort with a different scanner (catalina T2*, 75 cases, 830 CMB)?

## 2 Material
VALDO Task 2: 72 cases, 236 CMB; cohort 1 0.45x0.45x4 mm (11 cases), cohort 2 0.49x0.49x0.8 mm (34), cohort 3
1x1x3-4 mm (27). Split: 15 cases (97 CMB) set aside, 57 cases (139 CMB) in 5 stratified folds; the inner fold
selects epoch/threshold. catalina: T2* 75/830, SWI 61/653; CMB contrast 0.30 (VALDO 0.46-0.51), slice 5 mm.

## 2a State of the art: two lines that do not know each other
MicrobleedNet (preprint 11/2021, paper 2023) and the VALDO challenge (MICCAI 2021, paper 2024) arose in parallel.
The MicrobleedNet paper uses 20 VALDO cases only to tune hyperparameters and does not compare itself with any
VALDO team; conversely, no VALDO team used FRST, vessel removal, or distillation. The VALDO leaders won with
long training, large 3D context, and three modalities; MicrobleedNet relies on prior knowledge (FRST, vessel
mask, shape filter, edge distance) and a second stage. This work tests both families of ideas on the same data.
Details and sources: `research/sundaresan-vs-valdo.md`, `research/valdo-challenge.md`.
The original paper itself names two limitations that are measured here: vessel removal can remove CMB
(in our case 109 of 236), rotation/scaling can destroy CMB (our negative augmentation result).

## 3 Method of investigation
- Metrics: pooled lesion F1 (26-connectivity components, touching) as the ranking metric; precision, sensitivity, FP per case;
  in addition the official VALDO rule (median F1 per subject, 5 mm) for comparison against the challenge.
- Comparisons are always PAIRED over the same cases (bootstrap over cases, 95% interval).
- Noise level: same run with a different seed. In the 40-epoch recipe with early stopping around 0.08 F1 -- hence since
  19 Sep fixed 60 epochs for all comparisons. **Refinement 21 Sep (terminology check, section 11):** "fixed" means: the learning-rate schedule always runs for 60 epochs, there is no
  early stopping -- but the test prediction comes from the checkpoint with the best F1 of the INNER fold (evaluated every 5 epochs; `cv5.py`: `model.load_state_dict(best["state"])`), not from epoch 60.
  Re-measured in the new recipe (60 ep. fixed, without inpainting): seed 1 vs. 42 = 0.585 vs. 0.588, paired
  -0.003 [-0.066; +0.059]; at most 0.06 distance per fold. The recipe is stable; what is considered established is what leaves the
  paired interval of roughly +-0.06.
  **Addendum 22 Sep (third seed, 5u): the noise level had been set too low.** Three seeds of the identical recipe on gitter_d: 0.585 / 0.553 / 0.501 raw, 0.646 / 0.590 / 0.578 with the CSF rule --
  range 0.07-0.08, and seed 2 is paired "established" as worse against seed 42 (-0.083 [-0.135; -0.033]). The bootstrap over cases thus measures ONLY the uncertainty of case selection for ONE
  trained network, not that of training. Consequences for all comparisons in this document: (1) an effect from ONE seed per side counts as robust only from ~0.08 upward, below that it needs a second
  seed or an independent first stage (handled this way since 5j); (2) "established" individual findings below 0.08 (e.g. last-checkpoint s42 -0.047, catalina 120 ep. -0.022, blob -0.044) are directional,
  not proof -- precisely for that reason blob was confirmed with a second seed and the last-checkpoint finding with seed 1 was NOT; (3) effects above 0.08 (inpainting +0.136, 24 mm patch -0.216, no FRST
  -0.053 just below) remain robust. For the paper: report the baseline as mean +- range over three seeds (0.605 +- 0.034 with the rule), not 0.646.
- Grid searches only with cross-wise cell selection; among equivalent settings the simplest one wins.

## 4 Findings and interventions (each as a chain of argument)

### 4.0 Runnability of the published code (fork `custom_microbleednet`, our contribution)
- **Original:** commit 958f1cb. **Observation (read in the source code, not yet executed):** three calls in the
  fine-tuning path do not match the signatures -- `commands.py` passes `aug/save_cp/save_wei` to a `main()` that
  expects `perform_augmentation/save_checkpoint/save_weights`; `CDetPatchDataset` is built without the mandatory argument `ratio`;
  `CDiscPatchDataset` gets its classes swapped (negative, positive instead of minority, majority) and likewise
  no `ratio`. In addition, `freeze_layers_for_finetuning` always freezes the classifier head of CDisc.
- **Consequence (observed in the MB arena):** CDisc collapses to "all negative" during fine-tuning (accuracy 0.5).
- **Change in the fork:** calls fixed, positive:negative ratio fixed at 1:1, head (and `inpconv`)
  unfrozen, brain mask with filled holes, candidate threshold 0.2 -> 0.1 (on catalina-SWI, 0.2 yielded no candidates
  in 59 of 61 cases). Individually with effect size: `DEVIATIONS-microbleednet.md`, A1-A9.
- **Limit:** A1-A3 are read, not executed; before any outward-facing statement, re-verify on a fresh clone.

### 4.1 Vessel inpainting
- **Original:** `inpaint_vessels`: Frangi per slice (`beta=20`, scales in pixels), KMeans k=2 per slice, what is round and
  solid is preserved.
- **Observation:** affects 109 of 236 CMB, 27 completely, 17 invisible afterward; paints over 12-17 % of the brain; damage almost only
  at 3-4 mm slices (cohort 2: 0 invisible). FRST hit rate at the CMB 85 % with, 96 % without inpainting (recomputed 21 Sep with `cmb/analysis/frst_hit_rate.py`: 82 % / 96 %, see section 10).
- **Hypothesis:** `beta=20` turns off the suppression of round blobs; pixel-based scales and the forced cluster per
  slice make the procedure dependent on resolution and slice content.
- **Change:** inpainting 2 (`code/uebermalung2.py`): beta 0.5, scales in mm, one threshold per volume, only long
  thin components. Damage 1 of 236 with 0.9-1.5 % of the brain painted over; carries over unchanged to catalina-T2* (1/830)
  and -SWI (11/653 touched, 0 halved).
- **Experiment (running):** dose-response series none / 2%,12 mm / 3%,8 mm / 5%,6 mm / original, 60 epochs each;
  plus a grid search as a post-filter (50 cells, no training).
- **Result post-filter:** does not change F1 in any cell (0.5236): 0 of 37 predictions lie on long vessels.
- **Result training WITHOUT inpainting (gitter_d, 60 ep.; status 15:30 with 4 of 5 folds, final status 5/5 below):** F1 0.591 vs. 0.455 of the
  baseline on the same 46 cases, paired **+0.136 [+0.065; +0.226]**, established. Sensitivity 0.42 -> 0.57, precision
  0.50 -> 0.62, false positives per case 1.0 -> 0.9; per fold 0.65 0.50 0.63 0.56 vs. 0.48 0.35 0.48 0.48. The
  original inpainting thus costs about 0.14 F1 -- and omitting it does NOT bring more false positives, but fewer.
- **Final status 5/5 folds (16:00):** F1 0.588 (P 0.60, S 0.58, 0.93 FP per case); with the CSF rule 0.612 (P 0.66, S 0.57, 0.70).
  By challenge rule (median F1 per subject, 5 mm): 0.667 [0.11; 0.76] -- the same level as challenge places 1-3
  (0.667-0.684), though CV on 57 cases vs. a held-out test with 147, so no ranking comparison.
- **Inpainting 2 (3 %, 8 mm; gitter_e) vs. NONE, 60 ep., 5/5 folds (19 Sep 18:00):** F1 0.584 vs. 0.588, paired
  -0.005 [-0.065; +0.058] -- no difference. It only shifts the balance (precision 0.60 -> 0.64,
  sensitivity 0.58 -> 0.54, false positives per case 0.93 -> 0.75). Cautious stage 2%/12 mm (gitter_f): F1 0.588 vs. 0.588, paired -0.000 [-0.063; +0.055]. Strong stage 5%/6 mm (gitter_g): 0.570, paired -0.018 [-0.074; +0.028].
  **Complete dose-response:** none 0.588 | 2%/12 mm 0.588 | 3%/8 mm 0.584 | 5%/6 mm 0.570 | original 0.470 -- the more that is painted over, the worse; nowhere a gain.
- **Decision:** drop the original inpainting. Whether inpainting 2 (gitter_e/f/g) still adds anything over "none",
  the next three runs will show; after the post-filter result, little is to be expected.

### 4.2 Brain mask
- **Original / starting point:** mask = image > 0; VALDO delivers "masked".
- **Observation:** only cohort 1 is skull-stripped (1419 ml); cohort 2 1770 ml (16 % non-brain), cohort 3 2533 ml
  (40 %: bone, scalp, skull base). No CMB is cut off.
- **Hypothesis:** dark bone next to the brain creates confusion and distorts the intensity scale.
- **Change:** SynthSeg brain + 2 voxels (`--maske synthseg`), removes 19 % of the old mask, 0 of 139 CMB outside.
- **Result (60 ep., measured 19 Sep):** F1 0.421 vs. 0.470 of the baseline, paired -0.048 [-0.099; +0.009]; per fold
  0.47 0.32 0.36 0.44 0.55 vs. 0.48 0.35 0.48 0.48 0.53. No gain, rather a loss (not established).
- **Decision:** do NOT adopt the tight mask as the input mask. **Interpreted / open:** the mask was widened by only
  2 voxels (1 mm in-plane) -- the network may be missing context at the brain surface, where many
  lobar CMB are located; a 5 mm margin would be the fair second attempt. As a POST-FILTER the anatomy remains effective
  (CSF rule). **Lesson:** check "masked" twofold -- does it cut things off AND how much non-brain remains; and:
  a plausible repair is not yet a gain, only the measurement decides.

### 4.3 False positives in the CSF
- **Observation:** 98 of 176 false positives lie in the free sulcal CSF (SynthSeg label 24), not in the ventricles.
- **Change:** parameter-free rule "discard component with majority label CSF".
- **Result:** F1 0.245 -> 0.335 (attention U-Net), 0.404 -> 0.487 (a03, 40 ep.), 0.470 -> 0.524 (a03, 60 ep.); TP loss 0-2.
- **Side finding:** an earlier filter (we5) measured the opposite because its label table was wrong
  (left putamen/pallidum instead of ventricles). **Limits:** the rule was formulated after looking at the CV data; 5 of 139
  reference CMB lie in the CSF themselves.

### 4.4 Architecture
- **Original:** two-stage (candidates 48^3 + classifier 24^3, distillation).
- **Experiment:** tournament over nine single-stage 3D/2.5D networks in the same pipeline.
- **Result:** the only established gain: anisotropic U-Net (first stage in-plane only) +0.159 [+0.091; +0.234]
  vs. the attention U-Net; wins in all three cohorts. Deep/large networks (residual SE, SegResNet, Swin)
  established worse or not trainable in the shared recipe. Attention gates on the winner: no discernible
  gain (40 ep.; 60 ep. queued).
- **Interpreted:** with interpolated thick slices, it helps to trust only the measured direction first.
- **Done (19-21 Sep):** the two-stage version built on the winner has been built and measured -- 5d (per cohort, no gain) and 5j (shared across all cohorts, established better on SWI).

### 4.5 Training recipe
- **Observation:** second seed 0.321 instead of 0.404; cause a fold with early stopping at epoch 5.
- **Change:** fixed 60 epochs. **Result:** 0.404 -> 0.470 raw, all folds 0.35-0.53. 300 epochs queued.
- **Literature support:** the three best VALDO teams trained long (nnU-Net 1000 epochs), with large context and
  T1+T2+T2* (Sudre et al. 2024, arXiv 2208.07167; not checked line by line by me).

### 4.7 Intensity normalization + augmentation (negative result, 19 Sep)
- **Observation:** scale not harmonized between cohorts (input = 1 - image/maximum; 5th-95th percentile in the brain
  0.25-0.74 in cohort 1 vs. 0.65-0.98 in cohort 2); augmentation so far only mirroring.
- **Hypothesis:** per-case z-normalization and richer augmentation (rotation +-15 degrees, scale, contrast, noise)
  improve generalization.
- **Experiment:** a03-aniso, 60 fixed epochs, both TOGETHER as a package (`--norm z --aug`), paired against the baseline.
- **Result:** F1 0.410 vs. 0.470 (with CSF rule 0.472 vs. 0.524); paired -0.060 [-0.135; +0.011]; all five
  folds lower (0.43 0.33 0.40 0.43 0.45 vs. 0.48 0.35 0.48 0.48 0.53). Not established, but consistent.
- **Decision:** do not adopt the package. **Limits / open:** (a) the package does not separate whether normalization or
  augmentation is harmful; (b) the z-normalization is computed over the VALDO mask, which is 40 % non-brain in cohort 3 --
  a fair test needs the tight mask (`--maske synthseg --norm z`); (c) augmentation usually needs MORE
  epochs, 60 may be too few; the rotation interpolates the few voxels of a CMB (the VALDO paper warns of this).

### 4.8 Conversion to the common grid (finding 19 Sep, experiments running)
- **Starting point:** all cases are resampled to 0.5 x 0.5 x 1.0 mm (image linear, label nearest neighbor).
- **Observation (measured, 57 CV cases, data without inpainting):** the label volume is preserved (-1 / 0 / 0 %),
  but the contrast of the CMB against a ring of 0.5-2 mm in-plane drops: cohort 1 0.148 -> 0.114 (-23 %),
  cohort 2 0.123 -> 0.109 (-12 %), cohort 3 0.128 -> 0.087 (-32 %) [recomputed 21 Sep with `cmb/analysis/resampling_contrast.py`: -26 / -15 / -30 %, see section 10]. (A first attempt with a shell in
  voxels instead of mm gave -36 / -14 / -36 %, but was not comparable between grids and has been discarded.)
- **Hypothesis:** linear interpolation smears objects of only a few voxels; strongest where the most
  upsampling happens (cohort 3: 1 mm -> 0.5 mm in-plane and 3-4 mm -> 1 mm in z; median CMB there 4 voxels).
- **Experiments:** (a) grid with z = 2 mm instead of 1 mm (`dev/gitter_z2`, queued); (b) cubic instead of linear
  interpolation of the image (`GITTER_C_ORDNUNG=3`, re-measure contrast on the CPU first, then train).
- **Result z = 2 mm (20 Sep 02:40, 5/5 folds, patch 20x62x94):** F1 0.552 vs. 0.588 (z = 1 mm); paired -0.036 [-0.105; +0.032].
- **Decision per the pre-set rule: stays at z = 1 mm.** Less interpolation does not help; presumably
  the network needs the finer z sampling to distinguish "ends after one slice" from "continues". Caveat: a different
  grid size also means different label rasterization, the comparison is not voxel-identical. Cubic interpolation remains open.

### 4.9 Bias correction and intensity scale (experiments queued)
- Bias correction (FSL FAST -B, in our case before the pipeline): statistically has a clear effect in cohort 1, barely in cohort 2;
  its benefit for the F1 was never measured -> `dev/gitter_h` without bias correction, paired against gitter_d.
  Final status 20 Sep (5/5 folds): without bias correction 0.566 vs. 0.588, paired -0.022 [-0.097; +0.052] -- a small, not established
  disadvantage; the correction stays (it does not hurt, and for catalina it is already available upstream).
  Percentile scale p99.5 (5/5 folds): 0.613 vs. 0.588, paired +0.025 [-0.039; +0.092]; four of five folds better
  (0.67 0.49 0.61 0.65 0.65 vs. 0.65 0.50 0.63 0.56 0.58). Not established, but the only variant with a positive sign
  -> candidate for the final recipe, to be confirmed with a second seed. (The z-normalization, not this scale, was the ingredient in the discarded package 4.7.)
- Scale: the original divides by the maximum (sensitive to outliers) -> `--norm p995` divides by the 99.5th percentile in the
  brain; kept separate from the augmentation, which was measured together in package 4.7.

### 4.10 Additional channel: the target cohort decides
catalina has NO T1. FLAIR is locally available for only 17 of 75 T2* cases (23 of 61 SWI), 0.7 x 0.7 x 2.5 mm, not in
the T2* space (counted 19 Sep). The T1 channel of the VALDO leaders is therefore worthless for the transfer target.
What can be tested on VALDO is only T2 as a stand-in for FLAIR; that requires true multi-channel support and will
come with the `cmb/` package. Until then the transfer model stays with T2* + FRST -- both can be computed on any
cohort from the T2* alone. Confirmed by the user (19 Sep): in the catalina database, only a subgroup has FLAIR, the
rest only SWI or T2*. A model that requires FLAIR as a mandatory input would not be applicable to most of the
cohort -> the detection model remains single-sequence (T2* resp. SWI + FRST computed from it). FLAIR is at most a
candidate as an OPTIONAL channel with channel dropout (40 cases, benefit questionable) or for secondary analyses of
the subgroup (e.g. CMB burden vs. WMH burden).

### 4.11 Does FRST help? (interim status 19 Sep, decisive run brought forward)
- **Original:** FRST (radial symmetry) is a mandatory channel alongside the image in MicrobleedNet.
- **Measured so far:** (a) FRST reliably marks the bleeds: in 96 % the maximum lies in the top 1 % of brain voxels
  (without inpainting). (b) But it does NOT separate true from false candidates: AUC 0.56 (0.5 = worthless); FRST >= 0.9 is present in 96 %
  of the true and 88 % of the false candidates. For comparison: network probability and extent 0.69-0.70, contrast 0.65.
  (c) In the random forest of the second stage it contributes 0.01. (d) Old, 40 epochs, inpainted data: with FRST 0.215, without 0.175,
  not established.
- **Interpreted:** FRST responds to anything dark and round -- hence also to exactly the confusable candidates. As a pointer
  ("look here") it can still help the network; as a discriminating feature it is not useful.
- **Experiment:** a03-aniso, 60 ep., gitter_d, T2* only (`--kanaele t2s`), paired against the baseline with FRST; runs next.
- **Result (20 Sep 02:40, 5/5 folds):** without FRST F1 0.535 (P 0.52, S 0.55, 1.21 FP per case) vs. 0.588 with FRST; paired
  -0.053 [-0.116; +0.004]; lower in all five folds (0.63 0.44 0.54 0.54 0.52 vs. 0.65 0.50 0.63 0.56 0.58).
- **Decision per the pre-set rule (simpler variant only if delta >= -0.02): FRST STAYS.**
- **Interpreted:** FRST does not separate true from false candidates (AUC 0.56), but helps the network as a pointer -- without it
  both sensitivity AND precision drop. This supports the channel from the original; for every new cohort FRST must therefore
  be computed (from the T2*/SWI alone, no additional image needed).

### 4.6 Checked and fine
Bias correction ran on 72/72; no CMB outside mask/crop; FRST location without offset (0.1 voxel; an apparent
offset of 0.9 voxel was a measurement artifact from argmax on a plateau).

## 5 Current performance status (VALDO CV, 57 cases, out-of-fold)
| Stage | F1 raw | F1 with CSF rule | Median F1 per subject (VALDO rule) |
|---|---|---|---|
| Starting point: 3D attention U-Net, 40 ep. | 0.245 | 0.335 | 0.21 / 0.32 |
| anisotropic U-Net, 40 ep. | 0.404 | 0.487 | 0.48 / 0.56 |
| anisotropic U-Net, 60 ep. fixed | 0.470 | 0.524 | still to compute |
| ... + z-normalization + augmentation | 0.410 | 0.472 | -- (discarded) |
| ... + tight SynthSeg mask | 0.421 | -- | -- (discarded) |
| ... WITHOUT vessel inpainting (5/5 folds) | 0.588 | 0.612 | 0.667 / 0.667 |
| ... + shared threshold and minimum size, chosen cross-wise | -- | 0.633 | still to compute |
| SHIVA-CMB without adaptation (knows 60 of 72 VALDO cases from training!) | 0.575 | -- | 0.667 |
| Challenge leaders (different test set) | -- | -- | 0.67-0.68 |

### 5c Error analysis of the best model (a03-aniso, 60 ep., without inpainting; 19 Sep evening)
Measured on the 201 candidates of the review page (approximation: "found" counts prediction components, not references):
- Of 38 cases with CMB, in 10 NOT A SINGLE ONE is found; 7 of these are cohort 3, almost all with only 1-2 CMB -- this is the
  lower quartile 0 of the challenge rule.
- Missed CMB (59): the highest network probability WITHIN the lesion is 0.12 in the median; in 37 % it is >= 0.2
  (the network "senses" them, the threshold cuts them away), in 39 % below 0.05 (the network is blind).
- By region: lobar weakest (roughly half found), deep and infratentorial roughly three quarters, mesiotemporal 0 of 3.
- False positives (53): 29 are tiny (< 5 mm3), 13 lie in the CSF, 31 come from cohort 2.
**Grid search threshold x minimum size on the stored probability maps (with CSF rule, without training):**
starting point (threshold of the inner fold, no minimum size) F1 0.613; best cell threshold 0.3 / 2 mm3: 0.646 (P 0.63, S 0.66);
**chosen cross-wise (cell determined on four folds, measured on the fifth): 0.633** (P 0.63, S 0.64, 0.93 FP per case).
Interpreted: the inner fold (~11 cases) picks thresholds that are too high (0.5-0.9); ONE shared, lower threshold
plus a minimum size in mm3 is better. The bigger lever behind this: candidates at low threshold + second stage.

### 5d Second stage (candidates + classifier) -- negative result, 19 Sep evening
- **Original / literature:** MicrobleedNet and VALDO team Zihao are two-stage; the 2026 cascade removes 77 % of the FP this way.
- **Hypothesis (from 5c):** candidates at low threshold catch the "sensed" bleeds, a classifier sorts out the FP.
- **Setup (`cmb/stage2/`):** out-of-fold maps of the best network, threshold 0.15, CSF rule, >= 1 mm3 -> 203 candidates
  (103 on a bleed, 100 false); they reach 100 of 139 reference bleeds (upper bound sensitivity 0.72, F1 with a
  perfect classifier 0.84). Nested evaluation: the classifier for fold k never sees fold k, its threshold is
  determined in an inner 4-fold CV of the training folds.
- **Result (pooled F1, 57 cases):** accept all candidates 0.590 | threshold on the stage-1 probability only
  (nested selection) **0.627** | logistic regression on 10 features 0.616 | random forest 0.579 | small 3D ResNet on
  32x32x16 patches, 3 seeds averaged 0.599 (precision 0.69, sensitivity 0.53).
- **Decision:** no second stage on VALDO alone. **Interpreted:** roughly 160 training candidates are not enough; the
  published cascades had thousands (11,424 in the 2026 paper). The most important features of the forest are the
  stage-1 probabilities themselves -- the large network has already evaluated the context. FRST and edge distance contribute nothing.
  **Open:** with catalina (830 bleeds) the number of candidates would be five times as large; re-check then.

### 5a External benchmark: SHIVA-CMB without adaptation (19 Sep)
Freely available 3D ResU-Net (Tsuchida et al. 2024; weights v2, three folds, CC BY-NC-SA 4.0), wrapper
`cmb/baselines/shiva_cmb.py`, 6 s per case on the CPU. On our 57 CV cases (threshold 0.5):
F1 0.575 (P 0.61, S 0.54, 0.83 FP per case); challenge rule median F1 per subject 0.667 [0.00; 0.97].
Per cohort: 1 (4 mm) F1 0.39, S 0.25 | 2 (0.8 mm) F1 0.61, S 0.69, 1.7 FP per case | 3 (3-4 mm) F1 0.62, P 1.00, S 0.44.
**IMPORTANT (read in the SHIVA paper): SHIVA-CMB was trained ON VALDO** -- 60 of the 72 public cases (SABRE 8, RSS 28,
ALFA 24) in training, 12 in test. The 0.575 is therefore NOT an independent comparison: the model knows most of these
cases. That our network, in true cross-validation (every case unseen), reaches 0.588, is strong against this backdrop.
The fair comparison is catalina-T2*, which neither SHIVA nor our network has seen. **Result there (75 cases, 830 CMB,
threshold 0.5): F1 0.312, precision 0.92, sensitivity 0.19, 0.17 false positives per case; median F1 per subject 0.42.**
From 0.575 to 0.312: that is the domain drop of a broadly trained model (450 scans, six cohorts) on a
foreign cohort with weaker contrast, smaller CMB, and 5 mm slices -- and the bar that our transfer model
must beat without adaptation. (Lower threshold not tested.)
Common weakness: cohort 1 (4 mm slices) -- SHIVA too finds only a quarter of the bleeds there.

### 5b Expert review of the errors (tool ready, judgment pending)
`cmb/review/build_review.py` builds an offline review page modeled on the WMH editor `7_qc_edit.py` (one
HTML file, no server, judgments as JSON, keyboard operation): per candidate seven consecutive axial slices
plus coronal and sagittal, judgment bleed / none / uncertain and, for "none", the reason (vessel, calcification,
sulcal CSF, artifact, edge). For a03-aniso-e60 without inpainting: 201 candidates = 53 false positives, 59 missed
reference bleeds, 89 hits (`ergebnisse/review/a03-aniso-e60-gd/review.html`). Analysis:
`cmb/review/summarise_review.py`. Goal: how much of the remaining gap is annotation noise?
**Blinded version for the label ceiling (21 Sep evening, user approval of recommendation 5q a).** *Observation:* the page from 19 Sep names, per candidate, the type ("false positive", "missed"), the network probability, and draws
network (red) and reference (yellow) -- good for exploring, but a judgment meant to measure the QUALITY of the REFERENCE must not know any of that. *Change:* `build_review.py --blind` (random order with `--seed`, neutral ids c001...,
no type / probability / size / outlines, instead a circle of 8 mm, identical for all, at the candidate location; the key is stored as `<out>.key.json` NEXT TO the page folder and is never loaded by the page), `--repeat N`
(N candidates appear twice -> the rater's agreement with themself) and `--fixed-cell` (candidates from the probability maps at the reported operating point 0.3 / 2 mm3 / CSF rule instead of from the per-fold-chosen
binary maps -- this tests exactly the errors behind the reported number). Default behavior unchanged (backups `*.vor-blind-0921`). *Built:* `ergebnisse/review/a03-aniso-e60-gd-blind/review.html`, 195 candidates = 54
false positives + 47 missed + 94 hit components (92 reference bleeds) + 20 repeats = 215 views; cross-check: 2 x 92 / (2 x 92 + 54 + 47) = 0.646 = the reported number. Leak check: the page contains neither type nor
probability nor outline files, the image names are neutral; rebuilding with the same seed is bit-identical. *Analysis, fixed BEFORE the first judgment* (`summarise_review.py --key`): the expert judgment replaces the reference SYMMETRICALLY --
false positive judged as bleed -> hit; missed reference judged as "no bleed" -> no longer counts; HIT reference judged as "no bleed" -> reference struck AND the hit becomes a false positive (so the re-judgment
can also hurt the network); "uncertain" and unjudged keep their status; for candidates shown twice, the FIRST view counts (timestamp). Output: F1 / P / S with the delivered and the re-judged reference,
bootstrap over the 57 cases, paired difference, agreement, and Cohen's kappa on the repeats. Sanity check with artificial judgments: empty 0.646; all false positives = bleed 0.861 (TP 146, FP 0); all missed = none 0.773;
all hits = none 0.000 (FP 148) -- checked by hand, correct. These are at the same time the outer bounds of what the judgment can move. *Limits:* a rater who is also a project participant (the blinding mitigates this,
does not remove it); shown is only what the network OR the reference found -- what both miss remains invisible, the re-judged sensitivity is an upper bound. **Status: awaiting the user's judgment.**

## 5e Transfer to the own cohort: experiment plan (19 Sep evening, runs queued)
**Question:** Which strategy delivers the best model on catalina (T2* 75 cases / 830 CMB; SWI 61 / 653) -- and what does it cost on VALDO?
**Common basis (from this day's findings):** anisotropic U-Net, 60 fixed epochs, T2*/SWI + FRST, NO
vessel inpainting, same grid 0.5 x 0.5 x 1.0 mm (`cmb/transfer/build_private_grids.py`: 75/75 and 61/61 cases built, 1 case each with
changed lesion count and 1 with lesion outside the crop). Folds: the fixed folds of the MB arena, stratified by CMB
count (classes 0 / 1-2 / 3-5 / >5; per fold 5-6 cases without CMB and 5-6 with > 5) resp. the VALDO folds.
**Strategies (all 5-fold, every test case unseen):**
| Abbrev. | Training | measures |
|---|---|---|
| zero-shot | VALDO only (ensemble of the 5 fold models) | domain drop without any adaptation; bar: SHIVA-CMB 0.312 |
| cat | catalina only | what the own cohort alone yields (control) |
| ft | VALDO fold model k as start, fine-tuning on catalina folds != k (30 ep., learning rate 1e-4) | does pretraining help at all? |
| mix | VALDO + catalina together, test = fold k of both cohorts | one model for everything; effect on VALDO |
| mix-alle | VALDO + T2* + SWI | does the other sequence help or hurt? |
**Evaluation (`cmb/transfer/evaluate.py`):** for ALL runs the same fixed post-processing (threshold 0.3, >= 2 mm3; on VALDO
chosen cross-wise, on catalina NOT re-tuned), pooled F1, precision, sensitivity, FP per case, median F1 per
subject, F1 per CMB-count class, FP in cases without CMB per cohort. Comparison paired per case. Sums of the private cohort are in mb-arena/.
**Expectation (recorded in advance):** mix >= ft > cat > zero-shot on catalina; mix costs nothing on VALDO or helps.
Smoke test zero-shot on the two most CMB-rich T2* cases: F1 0.71 (P 0.91, S 0.58) -- only a functional test, no claim.
**First result, zero-shot on catalina-T2* (75 cases, 828 CMB, fixed post-processing, without CSF rule):** F1 0.622, precision 0.60,
sensitivity 0.65, median F1 per subject 0.62 -- but 4.76 false positives per case (3.38 in bleed-free cases; on VALDO 1.5).
By CMB count: 1-2: 0.38 | 3-5: 0.54 | > 5: 0.68. SHIVA-CMB without adaptation: 0.312 (P 0.92, S 0.19; different post-processing).
Our network, trained on 57 VALDO cases, thus transfers markedly better than the foreign model trained on 450 scans;
the open problem on catalina is the false positives, not the sensitivity.
**Order (reordered 19 Sep 22:40):** first the two decisions that affect ALL transfer runs -- channels (with/without FRST)
and grid (z = 1 or 2 mm) --, decision rule set in advance: the simpler variant wins at paired delta >= -0.02.
The strategy comparison deliberately runs with 60 epochs; only the winning strategy gets the long training at the end.
**Results T2* (fixed post-processing 0.3 / 2 mm3, without CSF rule; 75 cases, 828 CMB), status 20 Sep 06:45:**
| Strategy | F1 | Precision | Sensitivity | FP per case | FP in bleed-free cases | Median F1 per subject | F1 at 1-2 / 3-5 / > 5 CMB |
|---|---|---|---|---|---|---|---|
| SHIVA-CMB without adaptation (own post-processing) | 0.312 | 0.92 | 0.19 | 0.17 | -- | 0.42 | -- |
| zero-shot (VALDO only, 57 cases) | 0.622 | 0.60 | 0.65 | 4.76 | 3.38 | 0.62 | 0.38 / 0.54 / 0.68 |
| catalina only (5-fold) | 0.663 | 0.67 | 0.66 | 3.63 | 1.73 | 0.67 | 0.37 / 0.60 / 0.70 |
| mixed VALDO + catalina (60 ep.) | 0.623 | 0.56 | 0.70 | 6.07 | 3.88 | 0.60 | 0.36 / 0.55 / 0.69 |
| fine-tuning (VALDO start, 30 ep., learning rate 1e-4) | 0.657 | 0.63 | 0.69 | 4.43 | 2.92 | 0.67 | 0.47 / 0.57 / 0.71 |
Mixed, VALDO part of the same runs: F1 0.560 (P 0.52, S 0.60, 1.35 FP per case) vs. 0.585 with pure VALDO training.
**Prior expectation refuted (20 Sep 08:50):** mixed training at 60 epochs is on NEITHER of the two cohorts better than
training on the cohort alone (catalina 0.623 vs. 0.663; VALDO 0.560 vs. 0.585). It finds more (sensitivity 0.70), but reports
markedly more false ones. Possible cause, not yet checked: the epoch is fixed at 800 patches -- with 132 instead of 57 or 75
cases, the network sees each case only half as often; a fair comparison would need equally many passes PER CASE.
**Paired against "catalina only" (75 cases, bootstrap over cases, 10,000 draws; `cmb/transfer/evaluate.py`):** zero-shot -0.041
[-0.069; -0.016] (established worse) | mixed -0.040 [-0.077; -0.012] (established worse) | fine-tuning -0.006 [-0.029; +0.015]
(on par). **Result T2*:** own training and fine-tuning are equally good; fine-tuning needs HALF the training time (30 instead of 60
epochs) and is better on cases with only 1-2 bleeds (0.47 vs. 0.37), but makes more false positives (4.4 vs. 3.6 per case).
Mixed training does not pay off at equal step count.
**SWI (61 cases, 652 CMB), status 20 Sep 11:50:** zero-shot F1 0.305 (P 0.32, S 0.29, 6.7 FP per case) -- the T2* network does NOT
transfer to SWI; SWI only 5-fold 0.634 (P 0.62, S 0.65, 4.3 FP per case, median 0.67; by CMB count 0.51 / 0.60 / 0.70).
**SWI fine-tuning (20 Sep 13:25):** F1 0.599 (P 0.62, S 0.58, 3.9 FP per case; by CMB count 0.50 / 0.57 / 0.65); paired against SWI-only
-0.036 [-0.074; +0.003], zero-shot -0.330 [-0.441; -0.220]. Unlike with T2* (on par there), the T2* starting point for SWI is more of
a burden than a help: same precision, but 7 points less sensitivity. Interpretation (hypothesis): 30 epochs at lr 1e-4 are not
enough to retrain a network tuned to the wrong contrast; the epoch question (5h) thus arises here in reverse.
File: mb-arena/transfer/auswertung_swi_alle_0920.json.
Interim conclusion: the pure VALDO network reaches 0.62 without a single catalina case; own training on catalina brings +0.04
and halves the false positives in bleed-free cases. Cases with only 1-2 bleeds remain weak in both (0.37-0.38).
**Decision 20 Sep 02:40:** FRST stays (-0.053 without), z = 1 mm stays (-0.036 with 2 mm) -> the transfer block runs UNCHANGED as queued.

**mix-alle (VALDO + catalina-T2* + catalina-SWI, 193 cases, 60 ep.; 20 Sep 17:00):** established worse in ALL three cohorts than the
respective individual training -- T2* 0.618 (paired -0.044 [-0.082; -0.018]), SWI 0.547 (-0.087 [-0.126; -0.048]), VALDO 0.434
(-0.150 [-0.238; -0.054]; for comparison mix-valdo-t2s on VALDO 0.560, -0.025 [-0.080; +0.040]). The more cohorts in the same
60 x 800 patches, the greater the loss, and the SMALLEST cohort loses the most (VALDO: 139 of 1619 bleeds, the
lesion-centered sampling draws it correspondingly rarely). This fits the budget hypothesis from 5h but does not prove it: a
true incompatibility of T2*/SWI would give the same picture. Only the run with equal budget per case (mix e120, queued) can separate this.
**Interim strategy conclusion (status 20 Sep):** one own model per sequence; T2* own training or fine-tuning (on par), SWI own
training; mixing does not pay off at a fixed budget.

### 5f Order recipe -> long training (user question 20 Sep)
The 300-epoch training (6 h) had been queued before the recipe ablations. Wrong order: if afterward a different
positive:negative ratio, attention, the percentile scale, or a larger patch wins, the long run would have to be repeated. New: first all
60-epoch ablations (1:2, 1:1, attention, large patches, percentile scale with a second seed), then fix the final recipe,
then ONE long training with it. The 300-epoch entry is deferred (marked as struck in the queue).

### 5g Positive:negative ratio as a chain (user request 20 Sep, rule set in advance)
After the runs 1:2 and 1:1, `cmb/analysis/ratio_chain.py` checks the results and autonomously queues the next step:
as long as the pooled F1 of the newest ratio exceeds the previous best, it continues to 1:3, 1:4, 1:5, at most 1:6;
at the first step without a gain, it stops. If neither 1:1 nor 1:2 helps, the opposite direction 2:1 is checked once. Every step
is written to the log with a paired interval; the curve as a whole is the result, not the individual step.
**Result 1:2 (21 Sep 06:40): the training COLLAPSES.** With 33 % lesion patches the network tips into the empty prediction: fold 0 inner F1 0.000 from epoch ~10 to 60, loss
sits at 0.178, test F1 0.091 (best model = epoch 5); fold 1 the same from epoch 15 (0.187 -> 0.178). The plateau value is exactly the loss of "predict nothing": Dice share 0.5 x share
of patches with a bleed (0.33) = 0.167 plus a small BCE remainder -- for empty patches, the smoothed Dice rewards the empty prediction with loss 0. At 60 % training escapes
this minimum, at 33 % it does not. Run aborted by me after two of five folds (two of two collapsed; the GPU goes to the tissue channel) -- folder `a03-aniso-e60-l33-gd` contains fold 0
complete, fold 1 incomplete. **Consequence:** fewer positives is not an option; the chain toward 1:3, 1:4 ... thus drops out on its own (rule: continue only if better). The 1:1 run remains
queued -- it shows where the edge between 60 % (stable) and 33 % (collapse) lies. `ratio_chain.py` no longer queues the opposite direction 2:1 automatically because of the incomplete 1:2 run;
if desired, do so by hand. **Side finding for the method:** the recipe is sensitive at this point -- the share of lesion patches is not a harmless knob, but keeps the training stable.
### 5h Does the choice of strategy depend on the epoch count? (User question, 20 Sep, midday)
**Observation** from the saved learning curves (`verlauf.json`, inner F1, mean of the 5 folds; T2*):
catalina-only e60: Ep. 30 0.624 -> 50 0.653 -> 60 0.657 (flat from 50 on). mixed e60: 30 0.602 -> 50 0.636 -> 55 0.648 -> 60 0.645
(trails by 0.02-0.03 from Ep. 25 to 45, catches up to 0.012 by the end). Fine-tuning e30: already 0.629 after 2 epochs, ~0.66 from Ep. 16 on.
**Limits of this observation:** (1) the inner fold of the mixed run also contains VALDO cases, so the curves are not
case-matched; (2) the cosine schedule drives the learning rate to zero, EVERY curve flattens at the end -- "flat" does not prove convergence;
(3) inner F1 with training threshold, not the fixed post-processing of the table.
**Hypothesis:** the lag of the mixed training is (partly) a budget effect: one epoch is a fixed 800 patches; with
57+75 instead of 75 cases the network sees each catalina case only ~0.57 times as often. Fine-tuning, by contrast, converges early and should
manage with fewer epochs.
**Experiment (queued 20 Sep, 12:20):** mixed e120 (same number of passes per catalina case as catalina-only e60), catalina-only
e120 as control (otherwise "more training" would be conflated with "mixing"), fine-tuning e10 (moved up, ~25 min). Evaluation as in 5e:
fixed post-processing, paired against catalina-only e60. **Decision rule set in advance:** mixing counts as rehabilitated only if
mixed e120, paired, is not worse than catalina-only e120 (not just than e60).

**User question, 21 Sep ("mix everything -- the dataset is then larger"): status and two new runs.** So far only mixing at the SAME budget (60 Ep. x 800 patches) has been measured:
VALDO + catalina-T2* (132 cases) -0.040 on catalina, -0.025 n.s. on VALDO; all three cohorts (193 cases) -0.044 (T2*), -0.087 (SWI), -0.150 (VALDO) -- the more cohorts, the worse, most
strongly for the smallest one. This could be due to the budget (each case is seen less often; the lesion-centred sample draws VALDO with only 9 % of the 139 of 1619 bleeds) or to genuine
incompatibility (different masks, slice thicknesses, SWI contrast, annotation style). Two ways to use the larger amount of data anyway: (a) same budget per case -- mixed with 120 Ep. against
own training with 120 Ep. (queued since 20 Sep, now moved up); (b) **first pretrain on EVERYTHING, then fine-tune on the target cohort** -- the usual way to make use of large mixed data:
start from the mix-alle fold models (193 cases), 30 Ep., lr 1e-4, once on catalina-T2*, once on catalina-SWI (there with `manifest_swi_mix.json`, so that the folds match mix-alle
and no test case was in the pretraining). Comparison paired against own training (T2* 0.663, SWI 0.634) and against fine-tuning from the pure VALDO network (0.657 / 0.599). **Expectation set in advance:**
(b) is better on SWI than fine-tuning from the T2* network (the pretraining now knows SWI); whether it beats own training is open.
**Result (b) on catalina-T2* (21 Sep, 10:00):** fine-tuning from the all-model F1 0.623 (P 0.58, S 0.67, 5.3 false positives per case) -- confirmed WORSE than own training (-0.039 [-0.083; -0.007]) and than
fine-tuning from the pure VALDO network (-0.033 [-0.066; -0.009]). The mixed pretraining is thus the worse starting point for T2*; 30 epochs at lr 1e-4 do not make up the lag of the all-model (0.618).
The SWI run (where the expectation was positive) is still pending.
**Result (b) on catalina-SWI (21 Sep, 13:50):** fine-tuning from the all-model F1 0.590 (P 0.58, S 0.60, 4.7 false positives per case) -- confirmed worse than own SWI training (-0.045 [-0.073; -0.021]) and not better
than fine-tuning from the pure T2* network (0.599). **My expectation set in advance was wrong:** the pretraining already knowing SWI does not help. The picture is thus the same for both sequences -- mixed pretraining is the
worse starting point; this fits with 5n (what is NOT a bleed does not transfer between the cohorts). What remains open is only the budget question (mixed 120 Ep. against own training 120 Ep., queued).
**Result (a), first part -- mixed with 120 epochs (21 Sep, 19:30; chosen checkpoints 100 / 100 / 90 / 110 / 100):** catalina-T2* F1 0.641 (P 0.58, S 0.71, 5.6 false positives per case) against mixed 60 Ep. 0.623:
paired **+0.019 [+0.003; +0.037], confirmed** -- the user's budget hypothesis is right in direction. Against own training with 60 Ep. (0.663): -0.021 [-0.054; +0.006], no longer confirmed worse (with 60 Ep. it was
-0.040, confirmed). VALDO share: 0.582 against VALDO-only 0.585, paired -0.003 [-0.066; +0.056] (with 60 Ep. -0.025). So mixing catches up with double the budget: level on VALDO, still 0.02 behind on catalina, mainly
via the false positives (5.6 against 3.6 per case). **Not yet decided:** per the rule set in advance, the comparison against own training with LIKEWISE 120 epochs (queued) is what counts -- does that also benefit from the length,
does the gap remain.
**Result (a), second part and DECISION -- own training with 120 epochs (22 Sep, 04:09; chosen checkpoints 70 / 60 / 40 / 90 / 120; `mb-arena/transfer/auswertung_t2star_cat_e120_0922.json`,
`auswertung_t2star_mix_vs_cat_e120_0922.json`, `operating_point_t2star_cat_e120_0922.json`):**
| catalina-T2*, fixed post-processing | F1 | P | S | FP per case | paired against own training 60 Ep. |
|---|---|---|---|---|---|
| own training 60 Ep. | **0.663** | 0.67 | 0.66 | 3.6 | -- |
| own training 120 Ep. | 0.640 | 0.60 | 0.69 | 5.0 | **-0.022 [-0.052; -0.002], confirmed worse** |
| mixed 120 Ep. | 0.641 | 0.58 | 0.71 | 5.6 | -0.021 [-0.054; +0.006] |
| mixed 60 Ep. | 0.623 | 0.56 | 0.70 | 6.1 | -0.040 [-0.077; -0.012] |
Comparison set in advance: mixed 120 against own 120 = **+0.001 [-0.021; +0.023] -- level.** With the CSF rule: own 120 Ep. 0.656 against 0.675 (cross-wise -0.016 [-0.033; +0.000]).
**Decision per the rule:** mixing is rehabilitated at the SAME BUDGET PER CASE -- it then costs nothing more. This does NOT change the best strategy: own training with 60 epochs stays in front (0.663 / 0.675); the mixed network
sits -0.021 (n.s.) below it and is level on VALDO (0.582 against 0.585). A single model for both cohorts is thus possible, at a cost of ~0.02 on catalina, not confirmed.
**What the four runs together say -- it is about the budget per case, not epochs:** one epoch is a fixed 800 patches, so a training case sees ~1070 patches with own training 60 Ep. (45 training cases per fold),
mixed 60 Ep. ~610 (79 cases), mixed 120 Ep. ~1210, own 120 Ep. ~2130. Ranking by F1: 1070 (0.663) > 1210 (0.641) ~ 2130 (0.640) > 610 (0.623). Too little budget costs (mixed 60), too much also costs (own 120: precision
0.67 -> 0.60, false positives 3.6 -> 5.0 -- the same pattern as the late checkpoints in 5o), and the checkpoint selection does not catch this because the cosine schedule is stretched with 120 Ep. (epoch 40 of 120 is a different state than
epoch 40 of 60). **My expectation from 20 Sep was half right:** the lag of mixing was a budget effect, but "more budget" is not a one-way street.
**Prediction for VALDO with 120 Ep. (running, recorded BEFORE the result):** VALDO already has ~1400 patches per training case with 60 Ep. (33-36 cases per fold), i.e. more than catalina; 120 Ep. = ~2800. Per the budget interpretation,
120 Ep. must NOT help there and will more likely hurt (more false positives). If this holds, "120 epochs in general?" is answered: no -- the right quantity is patches per case, ~1000-1400 in this recipe; more cases need more epochs,
more epochs with the same cases do not. If it does help after all, the budget interpretation is wrong and VALDO needs a different explanation (smaller cases, more background per patch).
**Result VALDO 120 Ep. (22 Sep, 06:50; chosen checkpoints 100 / 100 / 90 / 110 / 90; `ergebnisse/analysis/valdo_e120_gepoolt.json`, `valdo_e120_split.json`, `operating_point_valdo_e120.json`): the prediction holds.**
Fixed cell raw 0.556 against 0.585 (paired -0.029 [-0.068; +0.019]), P 0.47 against 0.52, S 0.68 against 0.67, false positives per case 1.84 against 1.51; with the CSF rule 0.610 against 0.646, cross-wise 0.592 against 0.622 (-0.031 [-0.078; +0.018]).
Per cohort: 4 mm +0.022 (n = 8, n.s.), 0.8 mm **-0.046 [-0.074; -0.014]**, 3-4 mm -0.015 (n.s.). On VALDO alone "worse" is not confirmed (57 cases; the gap is within the size of the seed noise), "not better" already is --
and the mechanism is the same as on catalina: more sensitivity, less precision, more false positives. **Answer to the user question "120 epochs better in general?": no.** Three paired measurements (mixed +0.019 confirmed, own
catalina -0.022 confirmed, VALDO -0.029 n.s.) are explained by ONE quantity -- patches per training case; 120 epochs help only where 60 were too few (mixed, ~610 per case), and hurt where 60 were already ~1000-1400 per case.
For "harder networks" (nnU-Net design, attention) this has not been measured; the challenger runs (5s) therefore deliberately run with 60 Ep. **Limits:** one seed per run; the rule "~1000-1400 patches per case" is read off four
points, not a fine optimum; the obvious counter-check "60 Ep. with 400 instead of 800 patches" (half budget), which would show whether VALDO too is already OVER the optimum, has not been tested.
**User question, 21 Sep, evening: "are 120 epochs better in general, especially with harder networks and more data?"** So far only the mixed run has been directly measured (+0.019 confirmed). Two clues from existing runs: (1) the
chosen checkpoint lies at the end of the schedule (>= epoch 55) more often, the more data is in the training -- VALDO 2 resp. 1 of 5 folds, catalina-T2* and -SWI each 3 of 5, mixed 4 of 5; whoever is still winning at the end of the schedule was not yet
finished. (2) an "epoch" with us is a fixed 800 patches; 60 epochs are 12,000 steps -- the VALDO leader (nnU-Net: 1000 x 250 steps) and CenSynCMB (200 epochs) train several times longer. What matters
is therefore the number of steps relative to the amount of data and network size, not the epoch count as such. **Open flank:** the architecture tournament (4.4) ran with 40 epochs and early stopping -- large networks (Residual-SE, U-Net++,
SegResNet, Swin) could have been disadvantaged by this; the ranking only holds "at this budget". **Queued:** base recipe on VALDO with 120 epochs (alongside the control run catalina-only 120 Ep.).
Expectation set in advance: little to no gain on VALDO alone (57 cases, checkpoints mostly before the end of the schedule), a small one on catalina.

**First result for 5h (20 Sep, 14:30): fine-tuning 10 against 30 epochs, T2*.** ft10 F1 0.619 (P 0.56, S 0.69, 6.0 FP per case), ft30 0.657
(P 0.63, S 0.69, 4.4); paired against catalina-only: ft10 -0.044 [-0.090; -0.003], ft30 -0.006 [-0.030; +0.015]. The sensitivity is already
there after 10 epochs, the additional epochs reduce FALSE POSITIVES. My reading of the learning curve ("practically done after 2 epochs") was
thus wrong -- the inner F1 with training threshold did not show the difference. Consequence: fine-tuning with 60 epochs queued (T2* and
SWI); what is open now is whether fine-tuning at the same epoch count OVERTAKES own training.

### 5i Where are the false positives and misses located? (Idea from the methods research, 20 Sep; `cmb/analysis/fp_location.py`)
**Motivation:** works on CMB detection reduce false positives using location knowledge (arXiv:2306.13020, there 14.7 -> 0.56 FP per subject, unverified).
So far we only use the CSF rule. **Measurement:** location per prediction and per reference bleed from the SynthSeg map (majority label),
fixed post-processing as in 5e. **VALDO (a03-aniso e60, 57 cases):**

| Location | Predictions | TP | FP | Precision | Reference | found | Sensitivity |
|---|---|---|---|---|---|---|---|
| lobar | 84 | 48 | 36 | 0.57 | 78 | 48 | 0.62 |
| deep | 44 | 33 | 11 | 0.75 | 38 | 31 | 0.82 |
| infratentorial | 19 | 13 | 6 | 0.68 | 15 | 13 | 0.87 |
| mesiotemporal | 0 | 0 | 0 | -- | 3 | 0 | 0.00 |
| CSF | 33 | 1 | 32 | 0.03 | 5 | 1 | 0.20 |

What-if (exploratory, NOT chosen cross-wise): removing CSF 0.585 -> 0.646 (known, 4.3); removing any other location or
applying a stricter threshold (0.5/0.7/0.9) WORSENS the F1. **catalina-T2* (sums; catalina-only / fine-tuning):** lobar precision
0.71 / 0.66, sensitivity 0.71 / 0.74; deep 0.67 / 0.68 and 0.79 / 0.79; **infratentorial precision 0.76 / 0.74, but sensitivity only
0.46 / 0.50** with 207 of 828 bleeds -- 111 of 283 misses (39 %) lie in the cerebellum and brainstem. The CSF rule only brings
+0.008 to +0.014 there (VALDO +0.06), because SynthSeg places 24 genuine bleeds into CSF on 5 mm T2*.
**Result:** location knowledge beyond the CSF rule does NOT reduce our false positives -- the false positives sit where the
bleeds also sit (lobar). The finding of the analysis is a different one: on catalina, infratentorial SENSITIVITY is the biggest
single gap. **Hypotheses:** 5 mm slices in the cerebellum (partial volume), dark neighbouring structures (dentate nucleus, sinus),
tight HD-BET mask at the lower edge; VALDO has only 15 infratentorial bleeds, the pretrained network barely knows the region.
**Next step:** break down missed infratentorial bleeds by size/contrast/edge distance (CPU), then possibly a
location-aware sample (draw infratentorial bleeds more often).

**CORRECTION (20 Sep, 17:10, `cmb/analysis/missed_lesions.py`): the "infratentorial gap" is a CASE effect, not a LOCATION effect.**
The statement above ("biggest single gap", hypotheses 5 mm slices / dentate nucleus / tight mask) was premature: I had read pooled
lesion counts without checking how they are distributed across cases. Re-measured (catalina-only, 75 cases, 828 bleeds):
- 25 cases carry infratentorial bleeds; the THREE cases with the most misses hold 128 of these 207 bleeds and
  95 of 111 misses (234 of all 828 bleeds together). In these three: infratentorial sensitivity 0.26 (80 of 128 smaller
  than 10 mm3), lobar 0.53. In the REMAINING 72 cases: infratentorial **0.80**, lobar 0.76, deep 0.79 -- no weak location.
- Mask hypothesis refuted: 0 infratentorial bleeds lie outside the brain mask, edge distance missed/found 11.0 / 12.7 mm.
- What distinguishes the misses is SIZE and CONTRAST: missed median 5.0 mm3 (roughly 2 native voxels at 0.69 x 0.69 x 5 mm)
  and contrast 0.137, found 19.4 mm3 and 0.241. Stratified by size x contrast (lobar | infratentorial): small/weak
  0.27 | 0.07, small/strong 0.73 | 0.31, medium 0.66-0.86 | 0.56-0.77, large 0.79-0.94 | 0.85-0.91. At lobar rates per stratum,
  infratentorial would be expected to be 0.61, observed is 0.46 -- the rest is contained in the small bleeds of the three cases.
- The network is not "close" on the misses: highest probability < 0.05 in 92 of 111 (infratentorial) and 107 of
  150 (lobar). A lower threshold only in the cerebellum/brainstem (0.2 / 0.1 / 0.05, exploratory) LOWERS the F1 (0.663 -> 0.660 / 0.650 /
  0.649; fine-tuning 0.657 -> 0.650 / 0.640 / 0.638). Structure: cerebellar cortex 0.35, cerebellar white matter 0.57, brainstem 0.43 (shaped by the three cases).
**Decision:** location-aware sampling dropped. The lever is the class "small and low-contrast" (176 of 828 bleeds, sensitivity
0.27 / 0.07) -- that is exactly where the lesion-wise loss (5j) applies. **Open question for the expert:** are the many 1-2-voxel markings
in the cerebellum of these three cases reliable microbleeds? (Case identifiers, private only: mb-arena/transfer/missed_lesions_t2star_0920_PRIVAT_faelle.json.)
**Counter-check via the QC page (17:15):** the private control page (mb-arena/qc/t2star-nur-catalina, 75 cases, sum TP 545 / FP 272 /
FN 283 = exactly the evaluation) shows 112 / 78 / 44 reference bleeds with 73 / 61 / 11 misses for the three cases -- TWO patients carry
134 of the 283 misses (47 %). Without the three cases: sensitivity 0.77 instead of 0.66, precision 0.64, F1 0.696 instead of 0.663 (72 cases, 594 bleeds).
Both numbers belong in a report; which is the headline number depends on the expert's judgement of the smallest markings.
**Lesson for the method:** pooled lesion mass on catalina is carried by a few cases with very many bleeds (3 cases = 28 %
of all bleeds); any breakdown by location, size, or similar needs the counter-check "without the largest cases".
</content>
### 5j Lesion-wise loss (blob loss), queued 20 Sep
**Original/starting point:** BCE + Dice over the whole patch. **Observation:** the class with 1-2 microbleeds is the weakest on all cohorts
(0.37-0.51), small microbleeds get lost in the global Dice. **Change:** `--verlust blob` in `code/turnier.py`
(Kofler et al., MICCAI 2023, arXiv:2205.08209): global + 0.5 x mean of the per-lesion losses, each WITHOUT the voxels of the other
lesions (corresponds to alpha=2, beta=1); 26-connectivity components, float32, blob term only on the main head. **Check (CPU):** empty patch ==
global loss; perfect prediction -> 0; one missed 8-voxel microbleed next to a 72-voxel microbleed costs 0.091 instead of 0.014.
**Experiment:** a03-aniso, 60 epochs, gitter_d, otherwise unchanged; comparison paired against the 0.588 baseline, additionally sensitivity by
lesion size. The logged loss value is NOT comparable with other runs (different scale), the internal F1 is.
**Interim status 20 Sep 17:55 (4 of 5 folds, fixed post-processing, paired on the same 46 cases):** blob - baseline = -0.057 [-0.132; +0.003].
The loss does what it's supposed to -- and pays for it: sensitivity 0.67 -> 0.70 (TP 74 -> 81 of 115), but false positives 58 -> 103. By fold
(baseline | blob, TP/FP/FN): F0 23/22/9 | 26/28/6; F1 15/17/11 | 14/16/12; F2 26/13/13 | 26/12/13; F3 10/6/8 | 15/47/3. Fold 3 carries the
loss: 47 false positives, spread across many cases (9, 7, 6, 4, ...), 35 of them with p >= 0.5 -- so not a threshold problem, this fold's model
fires more overall. User note (justified): in 2 of 4 folds blob is slightly ahead; +0.02 in one fold is however ~1 microbleed at 18-39
microbleeds per fold, the noise level per fold is 0.06. **Next steps (user request: repetitions, VALDO AND catalina):**
blob on catalina-T2* (828 microbleeds, paired interval there around +-0.025 instead of +-0.06 -- one run says more than several VALDO repetitions)
and blob with seed 1 on VALDO (baseline with seed 1 is available: 0.585); further seeds staggered if the picture remains unclear.
For a fair comparison additionally: re-select threshold/minimum size for the blob model cross-wise (the fixed 0.3 / 2 mm3 was chosen on the
baseline model).
**Final status VALDO 5/5 folds (20 Sep 18:05; `cmb.transfer.evaluate`, `cmb.analysis.missed_lesions`, `cmb.analysis.fp_location`):**
F1 0.540 vs. 0.585, paired -0.044 [-0.105; +0.003]; sensitivity 0.67 -> 0.72, precision 0.52 -> 0.43, false positives per case 1.51 -> 2.30 (86 -> 131).
Sensitivity by size (baseline | blob): small < 7.2 mm3 (n=44) 0.52 | **0.64**, medium (n=48) 0.69 | 0.73, large (n=47) 0.79 | 0.79; contrast below
median 0.51 | 0.55, above 0.83 | 0.89. So the loss reaches exactly the microbleeds it was built for. 43 of the 131 false positives lie in CSF.
**Operating point per model, chosen cross-wise, with CSF rule (`cmb/analysis/operating_point.py`, new):** baseline fixed cell 0.646, cross-wise 0.622
(cells per fold all at threshold 0.3); blob fixed cell 0.599, best single cell 0.6 / 2 mm3 = 0.626 (optimistic), cross-wise **0.576**
(cells scatter: 0.5/4, 0.6/1, 0.6/2, 0.3/4, 0.4/4). Paired cross-wise -0.046 [-0.104; +0.018]. The blob model is calibrated differently (optimum at
0.5-0.6 instead of 0.3), but re-selecting doesn't save it -- the choice is unstable across folds.
**Second stage on the blob candidates (user 20 Sep: "at the first stage more false positives aren't a problem, the second is meant to remove them"):**
As a first stage blob is BETTER: candidates (threshold 0.15, CSF rule) reach 107 of 139 microbleeds (upper bound 0.77) against 100 (0.72) for the
baseline model; at threshold 0.10 / 0.05: 109 / 110 (0.78 / 0.79) with 339 / 371 candidates. 29 microbleeds (21%) are not reached by any candidate even at 0.05.
Nested evaluation (CPU classifiers): threshold only 0.608 | logistic regression **0.623** (P 0.63, S 0.62, 0.89 FP per case) | random forest 0.614;
at candidate threshold 0.05: 0.608 | 0.624 | 0.604. For the first time the second stage beats the plain threshold (+0.015; "contrast" is now the third most important
feature), the chain blob + stage 2 is thus ON PAR with the baseline chain (0.622-0.627), not above it: the second stage removes the false positives
(2.96 -> 0.89 per case), but in doing so loses the sensitivity gained (0.77 -> 0.62). The bottleneck is now stage 2 (115 positive training candidates).
**Counter-check catalina-T2*, baseline model catalina-only (20 Sep 18:40, CPU classifiers, nested):** 961 candidates (582 on one microbleed, 379 wrong)
reach 571 of 828 microbleeds -- upper bound only 0.69. Accept all 0.642 | threshold only **0.670** | logistic regression 0.658 | random forest 0.657. With FIVE TIMES as many
examples the second stage still does NOT beat the plain threshold. The interpretation from 19 Sep ("too few candidates") is thus not sufficient: hand-crafted features
see nothing that the first stage hasn't already accounted for. A second stage only pays off if it gets DIFFERENT information (larger context, other views,
agreement of several models) -- and the bigger lever on catalina is the upper bound of the first stage (31% of microbleeds are not reached by any candidate).
**Research on the second stage (20 Sep evening, `research/second-stage-fp-reduction.md`; one agent, light model, 23 calls -- over the budget of 15;
almost all numbers there from search snippets, marked as "unverified") and what of it was checked immediately:**
- *Gradient boosting on the same 10 features* (cheapest suggestion): VALDO-blob 0.609 (logistic 0.623, threshold only 0.608), catalina-baseline 0.655 (threshold only 0.670).
  No gain -- confirmed: it's not the type of classifier that's limiting, but the information content of the features.
- *Ensemble across seeds* (`cmb/analysis/ensemble.py`, new; mean of the out-of-fold maps, folds identical): baseline recipe, seeds 42 + 1 on VALDO. Fixed cell:
  0.594 vs. 0.585 / 0.553 of the members, false positives per case 1.30 vs. 1.51 / 1.42. With CSF rule and cross-wise chosen operating point: **0.647** (P 0.65, S 0.64,
  0.82 FP per case) vs. 0.622 / 0.564; paired against seed 42 +0.025 [-0.024; +0.079], against the mean of the members +0.054. Side effect: the operating point becomes
  STABLE (all five folds choose 0.3 / 2 mm3; the individual models scatter). First false-positive strategy with a consistently positive picture; not confirmed with only two members.
  Consequence: seed 2 queued for baseline and blob (at the same time the repetitions requested by the user).
- *Ensemble, continuation (20 Sep 19:15): MORE members do NOT continue the gain.* Built from existing runs of the same folds (seeds 42 and 1, percentile scale,
  three inpainting levels, blob). Fixed cell 0.3 / 2 mm3 with CSF rule -- individual models: s42 0.646, s1 0.590, np995 0.607, ge 0.610, gf 0.612, gg 0.591, blob 0.599
  (mean of the six baseline-close ones: 0.609). Ensembles: 2 members 0.647 | 3 0.648 | 5 0.630 | 6 0.639 | s42+blob 0.625 | s42+s1+blob 0.631 | all seven 0.644. So: ensemble ~
  best member, +0.02 to +0.04 above the MEAN member, flat from two members on; blob as a member brings nothing. Seed 42 is a lucky case (0.646 against
  0.59-0.61 of the others) -- the honest expectation for ONE model of this recipe is ~0.61, the ensemble reliably raises it to ~0.64. My wording from 18:55
  ("consistently positive picture") was too strong. The cross-wise chosen operating points (0.59-0.65) fluctuate more than the models themselves: at 57 cases, choosing
  among fewer than 40 cells per fold costs 0.02-0.05 F1 -> use the fixed, pre-selected cell for decisions. catalina-T2*: catalina-only + fine-tuning averaged 0.669 vs. 0.663,
  paired +0.007 [-0.007; +0.021]; with mixed as a third member 0.662. JSON: ergebnisse/analysis/operating_point_valdo_{ensembles,members}.json.
- *Not applicable to us:* focal loss / OHEM (our candidate sets are NOT imbalanced: 115:169 resp. 582:379), cascade of two classifiers (the second would have even
  fewer examples), phase/QSM (no data).
- *Open and promising, because they give the second stage DIFFERENT information:* mirror TTA (as an ensemble and as a consistency feature), context at multiple scales + location
  (Ghafoorian 2017, lacunes), pretrained encoders resp. 2.5D multi-view (Setio 2016), synthetic microbleeds AND synthetic hard negatives.
- **New state of the art, checked at the source (arXiv 2607.05325, 2026-07-06): CenSynCMB** (He, ..., Barkhof, Davies, Sudre -- the VALDO organizers): 3D attention U-Net +
  center maps as auxiliary output + reweighting by misses + fold-wise physics-based synthesis of positive CMB and marked hard negatives; VALDO Task 2
  lesion F1 74.3% ("best local-comparison", p = 0.020), external AIBL-SWI recall 88.5%, F1 65.0%. Assignment rule and split not yet read there -- clarify before a
  number comparison. New for us: center maps (related to the blob loss), reweighting by misses, synthesis of the NEGATIVES too.
  **Full text read (arXiv HTML, 20 Sep 19:20):** all 72 VALDO cases, 5-fold CV stratified by CMB yes/no, every case out-of-fold once (NOT the held-out test set);
  hit = predicted centroid <= 5 mm from the reference centroid, one-to-one (greedy); F1 PER SUBJECT, then averaged. Input T1 + T2 + T2* at 1 mm isotropic, percentile
  scale 0.5-99.5 + z-norm, patches 96^3 at ratio 2:1, 200 epochs, threshold 0.5, < 5 voxels removed. Table I: CenSynCMB R 75.3 / P 79.3 / F1 74.3, 0.8 FP per subject;
  attention U-Net (their backbone) 65.8; DynUNet 61.3; "FRST-style" 63.3; VALDO team Zihao 61.1; SwinUNETR 54.3. Ablation: backbone 65.8 -> + reweighting 67.6 -> + center map
  70.3 -> both 71.7 -> + synthetic CMB 73.1 -> + synthetic mimics 74.3. Center map: Gaussian per microbleed (sigma 2 voxels), focal MSE, weight via learned
  uncertainty. Reweighting: weight per training patch from microbleed size and current network response (small, weakly detected microbleeds count more). Synthesis: dipole kernel in
  k-space for the blooming, mimics as calcification-like and vessel-like sources, strictly fold-wise. No code published. **Conclusion:** the largest single contribution (+4.5
  points) is the CENTER MAP as auxiliary output -- cheap to reproduce; synthesis brings +2.6 at considerably more effort. Our numbers are only comparable with theirs once we
  compute the same rule (5 mm centroid, F1 averaged per subject) -- to be done.
  **Done (20 Sep 19:30, `cmb/analysis/subject_level_f1.py`, new; our 57 CV cases, T2* only, fixed post-processing + CSF rule):** F1 averaged per subject --
  seed 42: 0.571 (subjects with microbleeds only) resp. 0.626 (all; "nothing there, nothing reported" = 1), median 0.667, R 0.64 / P 0.62, 1.02 FP per subject, 14 of 19 microbleed-free
  subjects with no false positive. Seed 1: 0.463 / 0.519. Ensemble of two: 0.544 / 0.609 (P 0.64, 0.91 FP). blob: 0.486 / 0.517 (R 0.71, P 0.53, 1.61 FP). Attention U-Net from the
  tournament (40 epochs, with inpainting): 0.257. **Placement:** we are in the range of THEIR comparison methods (61-66), clearly below CenSynCMB (74.3; P 79, 0.8 FP per subject). The
  gap is in the PRECISION -- the per-subject measure punishes every false positive hard for subjects with 1-2 microbleeds, and that is exactly our weakest class. Not one-to-
  one comparable: they use all 72 cases and three sequences (T1, T2, T2*), we use 57 cases and only T2*; how microbleed-free subjects enter their average is not stated in the text.
- *Mirror TTA* (`cmb/analysis/tta_predict.py`, new; eight mirrored images, mean of the probabilities, still out-of-fold; CPU check: mirroring is exactly
  reverted): queued for VALDO-baseline, VALDO-blob and catalina-T2* catalina-only (~25-45 min GPU each), evaluation like the ensembles.
**Next steps:** 3D-ResNet stage on the blob candidates (queued), blob + stage 2 on catalina-T2* (5-6x as many candidates; `candidates.py`
has been able to handle manifests since today), ensemble across seeds, after that possibly blob weight 0.25 and hard negatives.

**blob on catalina-T2* (20 Sep 20:40; 75 cases, 828 microbleeds; evaluate, missed_lesions, operating_point, stage2):** fixed cell F1 0.634 vs. 0.663, paired
**-0.029 [-0.059; -0.007] -- confirmed worse**; sensitivity 0.66 -> 0.70, precision 0.67 -> 0.58, false positives per case 3.6 -> 5.7 (microbleed-free cases 1.7 -> 3.4). With CSF rule:
fixed cell 0.644 vs. 0.675, cross-wise 0.642 vs. 0.675, paired -0.032 [-0.060; -0.003]. **By size (catalina-only | blob): small < 10 mm3 (n=239) 0.27 | 0.28, medium (n=298)
0.72 | 0.81, large (n=291) 0.91 | 0.95;** contrast below median 0.46 | 0.52, above 0.86 | 0.90. Unlike on VALDO (small 0.52 -> 0.64), the loss does not reach the SMALL catalina microbleeds:
at 5 mm slices they are ~2 native voxels, the information is missing from the image -- the loss can only bring out what is there. Medium and large ones are gained, paid for with
false positives. By location the sensitivity rises slightly everywhere (lobar 0.71 -> 0.76, infratentorial 0.46 -> 0.52). **Second stage on the blob candidates:** 1135 candidates (605 positive,
530 wrong), upper bound 0.72 (baseline 0.69); threshold only 0.651 | logistic 0.653 | random forest 0.645 | boosting 0.629 -- even with 5-6x as many examples no gain, and the chain blob + stage 2
(0.653) stays BELOW the baseline with plain threshold (0.670). **Decision:** blob with weight 0.5 is not an improvement for catalina-T2*; the benefit on VALDO (small microbleeds) remains
a finding for thin slices -- the next sensible place to check is catalina-SWI (1.9 mm). Tool bug fixed: `missed_lesions.py` wrote truncated JSON files (numpy number
not serializable, key `cases` assigned twice); the printed tables were correct, all three JSON files regenerated.
**Mirror TTA on VALDO (20 Sep 20:40, eight views, still out-of-fold):** baseline 0.585 -> 0.604, paired +0.019 [-0.017; +0.058], false positives per case 1.51 -> 1.33 at the same
sensitivity; blob 0.540 -> 0.564, paired against blob **+0.024 [+0.001; +0.048]**, false positives 2.30 -> 2.07. With CSF rule fixed cell 0.646 -> 0.652 resp. 0.599 -> 0.613. In the per-
subject measure no gain (0.626 -> 0.621). Small, free, effect in the same direction for both models.
**Mirror TTA on catalina-T2* (20 Sep 20:50):** NO effect -- 0.663 -> 0.663, paired +0.000 [-0.012; +0.014]; with CSF rule 0.675 -> 0.671; false positives per case 3.63 -> 3.48. So:
TTA helps on VALDO (+0.02), not on catalina; it does no harm anywhere and remains an inference option, but is not a lever for the false positives on catalina.
**3D-ResNet as second stage on the VALDO blob candidates (20 Sep 20:45, GPU, nested, 3 seeds averaged):** 0.624 (P 0.66, S 0.59, 0.74 FP per case) -- on par with the
logistic regression (0.623), +0.016 over the plain threshold (0.608), on par with the baseline chain. So the CNN too sees nothing decisive that stage 1 hasn't already used.

**A common classifier for all cohorts (user idea 21 Sep: "the U-Nets are allowed to differ per cohort, the classifier should always see the same thing"; `cmb/stage2/pooled.py`, new).**
Each cohort keeps its first stage; the candidates of all cohorts (VALDO 203, catalina-T2* 961, catalina-SWI 870; together 2034, of which 1149 on one microbleed) are pooled, ONE classifier
learns on all of them. Nested evaluation as in 5d (classifier for fold k sees no candidates from fold k, regardless of cohort; acceptance threshold chosen internally per cohort; the seven
SWI twins of T2* subjects lie in their twin's fold). The script reproduces the earlier individual values exactly (VALDO 0.627 / 0.616 / 0.579, T2* 0.670 / 0.658).

| Method (pooled F1, nested) | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| threshold only on the stage-1 probability | 0.627 | 0.670 | 0.626 |
| own classifier per cohort (logistic / random forest) | 0.616 / 0.579 | 0.658 / 0.657 | 0.648 / 0.631 |
| COMMON classifier (logistic / random forest) | 0.608 / 0.612 | 0.670 / 0.664 | 0.652 / 0.653 |
| common + cohort ID (logistic) | 0.625 | 0.665 | 0.654 |

Paired against the plain threshold (bootstrap over cases): **SWI common logistic +0.026 [+0.013; +0.041], with ID +0.028 [+0.013; +0.043], random forest +0.027 [+0.008; +0.052] -- confirmed**; own
logistic +0.022 [+0.005; +0.043]. T2*: common -0.000 [-0.012; +0.011]. VALDO: common -0.020 [-0.051; +0.008]; the own random forest is confirmed worse there (-0.048), the common one no longer is.
**Measured:** (1) the idea holds direction -- the common classifier is at least as good as the own one in 5 of 6 comparisons, most clearly for the random forest on the smallest cohort
(VALDO 0.579 -> 0.612) and on SWI (0.631 -> 0.653); (2) for the first time a second stage beats the plain threshold CONFIRMED -- but only on SWI. **Caveat before interpreting:** SWI is the only cohort
without a CSF rule in the candidate filter (no tissue maps) -- the classifier there could just be catching up on what the rule already does in the other cohorts. **Counter-check running:** SynthSeg `--robust` on
all 61 SWI cases (CPU, `mb-arena/synthseg_swi.sh`, same settings as T2*), then SWI candidates WITH CSF rule and the same comparison.
**Counter-check and expansion (21 Sep 08:50).** SynthSeg `--robust` ran on all SWI cases (62/62, 0 errors). Finding: the HARD CSF rule doesn't fit SWI -- the map places 74 of the 652
reference microbleeds in CSF, 65 of which the network finds; the rule would delete them (F1 0.634 -> 0.608). The SWI gain of the classifier is therefore NOT catching up on the rule. Consequence:
candidates are now built WITHOUT the hard rule and carry the tissue class as a feature (`candidates.py --keep-csf`, field `tissue`); the common classifier gets it as five one-hot features
(`pooled.py --tissue`) and learns per cohort itself how much "lies in CSF" should count. In addition (user approval): candidates of the more sensitive networks (blob, center map) as TRAINING-ONLY
examples (`--extra`; the candidates of the baseline networks are still what is scored; for fold k only extra candidates from cases outside the fold).

| Method (nested, pooled F1) | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| threshold only, without rule | 0.565 | 0.654 | **0.626** |
| threshold only + hard CSF rule (previous chain) | **0.627** | **0.670** | 0.614 |
| common classifier without tissue feature (logistic / random forest) | 0.574 / 0.556 | 0.640 / 0.652 | 0.649 / 0.646 |
| common + tissue feature (logistic / random forest) | 0.574 / 0.624 | 0.668 / 0.679 | 0.651 / 0.650 |
| common + tissue feature + extra candidates (logistic / random forest) | 0.587 / **0.618** | 0.677 / **0.681** | **0.664** / **0.659** |

Paired against the RESPECTIVE best simple chain of the cohort (VALDO and T2*: threshold + hard rule; SWI: threshold without rule), random forest with tissue feature and extra candidates: VALDO -0.009 [-0.033; +0.019],
T2* +0.011 [-0.009; +0.036], **SWI +0.033 [+0.013; +0.057]** (logistic on SWI +0.039 [+0.017; +0.070]; precision 0.65 -> 0.74, false positives per case 3.5 -> 2.2 at the same sensitivity).
**Measured:** ONE classifier for all cohorts reaches the hard rule on VALDO and T2* (it learns it from the feature) and is confirmed better than everything so far on SWI; the extra candidates bring
another +0.002 to +0.014 per cohort. **Interpreted:** the second stage failed so far not because of the idea, but because its features contained nothing new -- the tissue class IS new, and a
common pool gives enough examples to weight it appropriately per cohort. **Limitations:** (1) many variants compared on the same cases -- the SWI gain is aligned across all ten variants
(+0.018 to +0.039), but the choice of the best variant is after the fact; as the final configuration it is fixed IN ADVANCE: random forest, tissue feature, extra candidates. (2) confirmation with an independent first stage
is still pending (SWI baseline with seed 1 queued). (3) the common 3D-ResNet (GPU) is queued.
**Sensitive first stage + common 3D-CNN (21 Sep 14:15, GPU ~80 min; blob candidates scored, baseline and center-map candidates as extra training):** VALDO 0.594 / with extra 0.607, catalina-T2* 0.673 / 0.682,
catalina-SWI 0.677 / 0.671. Paired over the same cases against the BASELINE stage with the same 3D-CNN (`cmb/stage2/compare_runs.py`): SWI +0.007 [-0.024; +0.034], T2* +0.000 [-0.016; +0.022], VALDO -0.036 [-0.117; +0.041].
Against the previous simple chain: SWI +0.051 [+0.013; +0.092], T2* +0.012 [-0.009; +0.040], VALDO -0.020 [-0.074; +0.046]. **Decision:** the sensitive first stage does not pay off even with the strongest classifier --
the gain comes from the second stage, not from the lesion-wise loss. Final chain thus: baseline-recipe U-Net per cohort -> common 3D-CNN (with extra candidates in training, which helps T2*).
**CONFIRMATION ATTEMPT with independent SWI stage (seed 1; 21 Sep 16:15) -- the final configuration fixed in advance is NOT confirmed.** First stage seed 1 on SWI: fixed cell 0.613 vs. 0.634
(paired -0.022 [-0.065; +0.012]); 893 soft candidates, upper bound 0.68. Common random forest + tissue feature + extra candidates: SWI 0.621 vs. plain threshold 0.617, paired **+0.004 [-0.021; +0.041]**; logistic -0.001
[-0.025; +0.031]. Precision rises as before (0.61 -> 0.68), but sensitivity falls (0.62 -> 0.57) -- net nothing. T2* unchanged +0.010 / +0.026 against rule / threshold. Side finding: VALDO drops in this run to
0.579 (previously 0.618), although ONLY the SWI candidates in the pool changed -- the VALDO value of the common classifier thus fluctuates by +-0.04 with the pool composition (278 candidates, threshold chosen on few cases).
**State of evidence:** the SWI gain of the feature classifiers (+0.033 / +0.039 with seed 42) is NOT reproduced with an independent first stage -- it therefore counts as unconfirmed. Still open is the 3D-CNN (largest gain,
+0.044): confirmation run on the seed-1 candidates queued (GPU, this evening). Until then, the simple chain continues to apply as the final chain: U-Net + fixed post-processing + hard CSF rule (VALDO, T2*) resp. without rule (SWI).
**CONFIRMATION of the 3D-CNN with independent SWI stage (21 Sep 20:00, GPU ~30 min):** on the candidates of the seed-1 network, the common 3D-CNN reaches SWI 0.662 (P 0.74, S 0.60, 2.25 false positives per case) against
plain threshold 0.617: paired **+0.045 [+0.018; +0.085]** -- practically the same gain as with seed 42 (+0.044 [+0.012; +0.085]). VALDO 0.636 (against the hard-rule chain +0.009 [-0.031; +0.054]), T2* 0.672 (+0.001).
**State of evidence, corrected:** the SWI gain of the 3D-CNN is reproduced with two independently trained first stages; the gain of the FEATURE classifiers (random forest, logistic) is not. This fits the interpretation from 5n: the CNN
sees the image around the candidate (vessel course, sulcus, CSF) and learns from it what is NOT a microbleed; ten numbers per candidate are not enough for that. **Decision:** final chain = baseline U-Net per cohort -> common 3D-CNN;
on VALDO and T2* it replaces the hard CSF rule equivalently (without needing a tissue map), on SWI it brings +0.045 and around two fewer false positives per case. Still open: the same confirmation for VALDO and T2* with the
respective second seeds (VALDO: candidates available, T2*: no second seed available) and the final measurement on the 15 held-out VALDO cases.
**Correction of naming (21 Sep 14:40, user question about the architecture):** the network called "3D-ResNet" in 5d and here is NOT a ResNet -- it has no residual connections. It is a small, plain 3D-CNN:
input 3 channels (T2* inverted, FRST, probability of the first stage), patch 16 x 32 x 32 voxels = 16 x 16 x 16 mm around the candidate; three double-convolution blocks (3x3x3, InstanceNorm, LeakyReLU) with 16 / 32 / 64
channels, in between two MaxPool 2 steps, then global average, dropout 0.3, one linear output; 216,081 parameters. Training: 40 epochs, AdamW 1e-3 (weight decay 1e-3), cosine, batch 16, BCE with class weight
n_neg / n_pos, augmentation by random offset (+-2 slices, +-4 voxels within the slice) and mirroring; three seeds averaged. In text and tables from now on "3D-CNN"; older entries remain unchanged.
**Common 3D-ResNet (21 Sep 10:20, GPU ~25 min; `pooled.py --models cnn`, soft candidates, WITHOUT tissue feature and without extra candidates):** VALDO 0.625 (P 0.65, S 0.60, 0.81 false positives per case),
catalina-T2* 0.672 (P 0.70, 3.0), **catalina-SWI 0.670** (P 0.74, S 0.61, 2.3). Paired against the best simple chain per cohort: VALDO -0.003 [-0.046; +0.038], T2* +0.001 [-0.014; +0.022],
**SWI +0.044 [+0.012; +0.085]**. The network sees no tissue map and still reaches the hard CSF rule on VALDO and T2* -- it recognizes the CSF from the image itself; on SWI it is the best
result so far (0.626 -> 0.670). For comparison: the cohort-own 3D-ResNet on VALDO stood at 0.599 (5d) -- the common pool helps the CNN too. Queued: the same with the extra candidates of the
sensitive networks (`--extra`).
**Two counter-checks on the CPU (21 Sep 11:10), before this turns into a plan change:** (a) SENSITIVE first stage (blob) as SCORED candidates, common random forest + tissue feature + extra training: VALDO 0.578,
T2* 0.671, SWI 0.655 -- not better than the baseline stage with the same classifier (0.618 / 0.681 / 0.659); the random forest does not realize the higher upper bound of the sensitive stage (0.79 / 0.74 / 0.73). Whether the
3D-ResNet can do it is checked by a queued GPU run. (b) VALDO with INDEPENDENT first stage (seed 1): hard-rule chain 0.559, common random forest 0.531 (-0.029 [-0.061; +0.011]), with extra training 0.507
(-0.053 [-0.110; +0.004]); T2* and SWI unchanged (+0.009 / +0.045 confirmed on SWI). **Status:** the gain of the second stage is robust on SWI (three classifier families, two pool compositions), small on T2*
and unconfirmed, not present on VALDO -- there the hard CSF rule (T1-based, precise map) remains at least equivalent. Tool: `cmb/stage2/compare_runs.py` for paired comparisons across
candidate sets (pooled.py now writes counts per case).
**3D-ResNet with extra candidates (21 Sep 12:50, GPU ~55 min):** VALDO 0.622, catalina-T2* **0.684** (P 0.72, S 0.65, 2.7 false positives per case; against the chain +0.014 [-0.002; +0.035]), SWI 0.669 (+0.044 against the
threshold, confirmed). The repeat run WITHOUT extra candidates in the same job gave 0.630 / 0.673 / 0.670 (first run 0.625 / 0.672 / 0.670) -- so the GPU variability of the CNN is ~0.005. The extra candidates
help only T2* (+0.011), nothing else. **Plan change (user approval 21 Sep 11:15):** further GPU time goes to the second stage and its confirmation; seven entries with U-Net variants without prospects are struck:
ratio 1:1 + chain (1:2 collapses, 5g), attention on the winner, percentile scale seed 1, fine-tuning 60 epochs (T2*, SWI), tissue channel on catalina (VALDO -0.058, 5m). Kept: 3D-ResNet on the candidates of the
sensitive first stage (running), SWI network seed 1 (independent confirmation), mixed / own with 120 epochs (budget question), small and large patch (context), baseline seed 2.

### 5k Center map as auxiliary output (after CenSynCMB 2026), queued 20 Sep evening
**Original / literature:** MicrobleedNet has no instance-related supervision; CenSynCMB (He et al., arXiv 2607.05325) introduces a center map as a second output channel --
the largest single contribution of their ablation (65.8 -> 70.3 F1 per subject). **Observation in our case:** our gap to CenSynCMB is in the precision (0.62 against 0.79); the
lesion-wise loss (5j) raises the sensitivity of small microbleeds, but also the false positives. **Hypothesis:** a head that must learn ONE Gaussian bump at the centroid per microbleed
forces the shared features toward "here is exactly one compact lesion" instead of "here it's dark" -- this should lower false positives at vessels and edges rather than raise them.
**Change (`code/turnier.py`, backup turnier.py.vor-zentrum-0920):** network `a12-anisozentrum` = a03-aniso + 1x1x1 head on the same decoder features; two
channels during training, in eval() only the segmentation (all tools remain usable unchanged; weights = a03 keys + `netz.zentrum.*`). Loss `--verlust zentrum` = BCE+Dice + focal MSE
on the center map, weight (C + eps)^2.5. **Deliberate deviations from the paper:** sigma 2 mm in mm instead of 2 voxels (our grid is 0.5 x 0.5 x 1 mm); normalized to the
weight sum with fixed weight 1 instead of learned uncertainty (our loss doesn't see the network parameters); eps 0.05 instead of 1e-3, so the head also learns "no center" in the
background (usable later as a second opinion for the second stage). **Check (CPU):** the bump sits at the centroid, 2 mm away 0.53-0.60 in x as in z; empty
patch -> map 0; gradient reaches the head and the shared features; eval() delivers one channel; save/load bit-identical; switch and network only permitted together.
**Experiment:** 60 epochs, gitter_d, seed 42, otherwise baseline recipe; comparison paired against the baseline at the fixed cell, plus the per-subject measure (5j) and false positives by location.
Not yet reproduced: reweighting by misses (+1.8 for them), synthesis (+2.6).
**Result (20 Sep 22:55, 5/5 folds without error, run time per fold like the baseline):** fixed cell F1 0.531 vs. 0.585, paired **-0.054 [-0.110; -0.006] -- confirmed worse**;
sensitivity 0.67 -> 0.73, precision 0.52 -> 0.42, false positives per case 1.51 -> 2.51 (86 -> 143; of which CSF 52, lobar 58, deep 20). With CSF rule: fixed cell 0.606 vs. 0.646, best
single cell 0.6 / 4 mm3 = 0.636 (optimistic), cross-wise 0.586 vs. 0.622 (-0.036 [-0.089; +0.020]). Per-subject measure (CenSynCMB protocol): 0.531 vs. 0.571 (subjects with microbleeds),
0.459 vs. 0.626 over all -- only 6 instead of 14 of the 19 microbleed-free subjects remain without a false positive. Sensitivity by size (baseline | center | blob): small 0.52 | 0.66 | 0.64, medium
0.69 | 0.69 | 0.73, large 0.79 | 0.85 | 0.79; low contrast 0.51 | 0.59 | 0.55.
**Hypothesis refuted:** the center map does NOT lower our false positives -- it acts like the lesion-wise loss: more sensitive (small microbleeds +14 points), calibrated differently
(optimum at threshold 0.6 instead of 0.3), more false positives, at its own best operating point at best on par. The gain from the paper (+4.5) does not transfer to our setup.
**Possible reasons (not checked):** (1) they evaluate at threshold 0.5 with checkpoint selection by validation F1, we at the fixed cell 0.3, which is unfavorable for instance-aware models;
(2) with T1 + T2 the additionally found mimics can be rejected, not with T2* alone -- more sensitivity then pulls in vessels and CSF artifacts too; (3) my
deviations (background term eps 0.05, fixed weight); (4) 60 instead of 200 epochs. **Decision:** record the negative result, NO second seed, no combination with fnrw.
**Second stage on the center-map candidates (user question 20 Sep 23:40 -- the numbers above are ONLY the first stage):** 275 candidates (117 positive, 158 wrong), upper bound 0.77;
nested: accept all 0.530 | threshold only on the own probability **0.635** (P 0.70, S 0.58, 0.61 FP per case) | logistic 0.595 | random forest 0.606. The second classifier
doesn't help here either; a higher, nested-chosen threshold brings the model to the level of the baseline chain (0.627), not above it.
**Cross-cutting finding from 5j + 5k:** both levers that make the network more instance-aware find more small microbleeds and pay for it with false positives; the F1 does not change for the better.
Our bottleneck is therefore not "too little sensitivity in training", but the SEPARATION of microbleed and mimic with only one sequence. Consequence: hard negatives (addresses the false positives)
moved ahead of fnrw (fnrw again emphasizes the missed ones and would likely show the same pattern); the next sensible combination would be blob/center + hard negatives.

### 5l Two more training levers against false positives and misses (built in 20 Sep evening, user approval)
**(a) Reweighting by misses, `--verlust fnrw`** (CenSynCMB, there +1.8 F1 points alone, +5.9 together with the center map): BCE+Dice per patch, weighted with
w = (mean size / size)^0.5 * (1 + 1.5 * (1 - highest current network response on the microbleed)), clamped to [0.5, 5]; patches without a microbleed weight 1. The paper does not say
how several microbleeds in one patch combine -- here the maximum. Difference to the lesion-wise loss (5j): this one weights by the network's CURRENT ability
(found microbleeds lose weight), blob weights rigidly by instance. CPU check: without microbleeds exactly cv5.loss_bce_dice; small missed microbleed weight 1.06 -> 1.00, as soon as
found; giant lesion 0.5; gradient finite.
**(b) Hard negatives, `--harte-negative`** (classic, for CMB Dou et al. 2016): from epoch 11, every 10 epochs the CURRENT network predicts its own training cases (fixed
post-processing); centroids of the false-positive components become centers for 60% of the microbleed-free patches. Strictly within the fold (no foreign run, no test case), the label
within the patch stays the true one. Cost: five mining rounds per fold (estimated +8 min). CPU check with an artificial false positive: before mining 1 of 22 microbleed-free patches at the
false positive, afterward 13 of 21; network back in training mode afterward; state reset per fold.
**Experiments (queued behind the center map):** blob on catalina-SWI (1.9 mm slices -- does the loss reach the small microbleeds there as on VALDO?), fnrw on VALDO,
hard negatives on VALDO; 60 epochs each, seed 42, paired against the baseline at the fixed cell, plus false positives per case and sensitivity by size. Struck: blob seed 2.
**Results (21 Sep 05:35, all runs 5/5 folds without error; fixed cell, paired against the baseline with seed 42):**

| Run | F1 | Precision | Sensitivity | False positives per case | Paired | With CSF rule (fixed) | F1 per subject | small / medium / large |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.585 | 0.52 | 0.67 | 1.51 | Reference | 0.646 | 0.571 | 0.52 / 0.69 / 0.79 |
| Hard negatives | 0.585 | 0.53 | 0.66 | 1.42 | +0.000 [-0.044; +0.048] | 0.625 | 0.546 | 0.52 / 0.67 / 0.77 |
| Reweighting (fnrw) | 0.555 | 0.46 | 0.69 | 1.95 | -0.030 [-0.080; +0.019] | 0.592 | 0.506 | 0.55 / 0.71 / 0.81 |

**Hard negatives: no effect.** The mining itself works (fold 4: 86 false-positive centers at epoch 31, 112 at epoch 41, 42 at epoch 51 -- so on the TRAINING cases the network does
learn its false positives away), but nothing changes on the test cases. **Interpretation (hypothesis):** the false positives are case-specific and get memorized, not generalized as a class
-- with 33 training cases per fold there are too few examples per mimic type. **fnrw:** same pattern as blob and center map, only weaker (more
sensitivity, more false positives, F1 not better). **Decision:** neither adopted; no combinations.
**blob on catalina-SWI (61 cases, 652 CMB; sums only):** F1 0.631 vs. 0.634, paired -0.003 [-0.036; +0.024] -- on par; sensitivity 0.65 -> 0.68, false positives per case 4.3 -> 5.0.
By size: small < 9.9 mm3 (n=217) **0.37 -> 0.44**, medium 0.70 -> 0.73, large 0.88 -> 0.86. The hypothesis from 5j holds: at thin slices (SWI 1.9 mm, VALDO) the loss reaches
the small microbleeds, at 5 mm T2* it doesn't -- it improves the F1 nowhere. (Tool: `missed_lesions.py` now also runs without a tissue map; for SWI there are no SynthSeg maps.)
**Overall balance of the training levers (5j-5l):** blob, center map, fnrw, hard negatives -- NONE improves the F1; three make the network more sensitive and pay for it with false positives, one
only works on the training cases. The baseline recipe stays. The finding supports the diagnosis from 5k: the bottleneck is the information in the image (one sequence, coarse slices), not the training signal.

**blob with second seed (21 Sep 06:20):** seed 1 blob 0.490 vs. baseline seed 1 0.553, paired -0.063 [-0.129; +0.006] (sensitivity 0.60 -> 0.70, false positives per case 1.42 -> 2.81).
Two-seed ensembles at the fixed cell with CSF rule: baseline 0.647, blob 0.604. The finding from 5j (more sensitive, more false positives, lower F1) repeats with the second
seed in the same direction and magnitude (-0.044 / -0.063) -- thereby concluded for VALDO.

### 5m Tissue map as input channels (user approval 21 Sep early; queued)
**Starting point:** no training lever improves the F1 (5j-5l); the bottleneck is the separation of microbleed / mimic with one sequence. FLAIR is ruled out for catalina (user 20 Sep: too few
cases, helps at most with post-processing). **Observation:** the only measure with "different information" that we have is the CSF rule from the SynthSeg map -- and it is
our largest single post-processing gain (VALDO +0.06). **Hypothesis:** if the network gets the map as INPUT, it learns the rule itself and, beyond that, location-dependent appearance
(sulci, ventricle border, basal ganglia), instead of us applying a single hard rule afterward. **Change (`code/turnier.py --gewebe [PATTERN]`, backup turnier.py.vor-gewebe-0921):**
five one-hot channels (CSF, cortex + mesiotemporal, white matter, deep nuclei, cerebellum + brainstem) in addition to T2* and FRST = 7 input channels; the map sits per case as ONE uint8 volume
in memory and is only fanned out within the patch. Sampling and sliding window are word-identical copies of the originals with the extra channels. **Check (CPU):** same random sequence ->
image, FRST, and label of the patches bit-identical with the baseline sampling (the comparison with the baseline is thus paired even in the drawn patches); one-hot correct; sliding window
bit-identical with the old inference; loader on VALDO (brain fractions: CSF 0.20, cortex 0.31, white matter 0.26, deep 0.03, infratentorial 0.10) and on catalina with conversion of the 1 mm map to the
grid (0.18 / 0.34 / 0.27 / 0.02 / 0.10); network with 7 channels computes forward and backward. **Experiments:** VALDO (`--gewebe`, map from T1 in T2* space) and catalina-T2* (map from
SynthSeg `--robust` directly on T2*), 60 epochs each, seed 42; evaluation at the fixed cell WITH and WITHOUT the CSF rule (if the network learns the rule itself, its added value shrinks), paired
against the baseline, false positives by location. **Caveat upfront:** on VALDO the map comes from the T1 -- a gain there demonstrates the value of ANATOMY, not the value of a T2*-own map;
decisive for the target cohort is the catalina run. SWI has no maps (not yet computed). Decision rule as always: below +0.03 not distinguishable from seed noise.
**Result VALDO (21 Sep 08:50, 5/5 folds, ~28 min per fold):** fixed cell F1 0.527 vs. 0.585, paired -0.058 [-0.138; +0.013]; precision 0.52 -> 0.43, false positives per case 1.5 -> 2.2.
The hypothesis holds true by half: the network does actually learn the CSF rule itself -- it makes NOT A SINGLE prediction in CSF anymore (0 instead of 32 false positives there; the rule then has nothing
left to do, with rule 0.527 vs. 0.646 of the baseline, paired -0.111 [-0.214; -0.007]). But outside the CSF the false positives rise (lobar 36 -> 81, deep 11 -> 32), and the 5 reference microbleeds with
CSF label are all lost. **Interpreted:** seven input channels at 57 cases -- the network relies on the map instead of the image. **Decision:** not adopted as U-Net INPUT.
The same information, by contrast, works as a FEATURE of the second stage (5j, common classifier): anatomy helps in deciding about candidates, not as a raw channel of a small training set.
The catalina run remains queued as announced (map there from T2* itself).
### 5n Why doesn't mixing help -- do the images differ, or the expert annotations? (user question 21 Sep; `cmb/analysis/cohort_differences.py`, new)
Measured on the common training grid, per cohort (catalina only as sums):

| Cohort | Cases / CMB | CMB per case (median / max) | Lesion in native voxels (median) | Share < 4 native voxels | Contrast (rel. darkening, median) | Brain median (inverted scale) | "mimic burden" |
|---|---|---|---|---|---|---|---|
| VALDO-1 (4 mm) | 8 / 32 | 3 / 12 | 15.5 | 0 % | 0.343 | 0.67 | 2.8 % |
| VALDO-2 (0.8 mm) | 27 / 80 | 0 / 26 | 38.0 | 0 % | 0.384 | 0.72 | 20.9 % |
| VALDO-3 (3-4 mm) | 22 / 27 | 1 / 2 | 5.3 | 41 % | 0.248 | 0.70 | 38.6 % |
| catalina-T2* (5 mm) | 75 / 828 | 3 / 112 | 6.3 | 29 % | 0.212 | 0.53 | 15.3 % |
| catalina-SWI (1.9 mm) | 61 / 652 | 1 / 102 | 21.3 | 5 % | 0.200 | 0.71 | 33.7 % |

(Mimic burden = share of brain voxels that are at least as bright as the cohort's mean bleed. An additionally computed "contour index" -- how much of the bleed brightness still sits in the first voxel ring OUTSIDE the mask -- is 0.21-0.32 for thin and 0.41 for thick slices; it follows the resolution, not the cohort, and is therefore not suited as a measure of annotation style.)
**Measured -- BOTH differ:** (1) Images: intensity level (catalina-T2* 0.53 vs ~0.70 -- tight HD-BET mask, different maximum), native voxel volume 0.19 to 3.0 mm3, mimic burden 3 to 39 %.
(2) Annotations: the catalina experts marked clearly WEAKER bleeds (darkening 20-21 % vs 34-38 % in VALDO-1/2) -- and this holds even on SWI with thin slices, so it is not just an effect of the 5-mm slices; on top of that, extreme cases with over 100 bleeds, which do not occur in VALDO.
**Cross-check whether the annotation style confuses the network:** the network trained ONLY on VALDO finds the same bleeds on catalina-T2* as the catalina network -- by size 0.29 / 0.71 / 0.87 vs 0.27 / 0.72 / 0.91, low-contrast 0.48 vs 0.46, high-contrast 0.82 vs 0.86. What makes it worse are the FALSE POSITIVES: 357 vs 272 (lobar 217 vs 156, CSF 95 vs 59). The mixed network: sensitivity higher in all classes (0.32 / 0.78 / 0.94), but 6.1 instead of 3.6 false positives per case.
**Interpreted:** what a bleed IS is learned the same way across cohorts by the networks -- the annotation style is no obstacle for sensitivity. What is NOT a bleed (vessel cross-sections, sulci, artifacts) looks different per scanner, sequence, and slice thickness and does not transfer; that is where mixing costs. This argues more strongly for IMAGE than for expert differences. It remains open how many of the additional "false positives" are real, unmarked bleeds -- only the expert review can clarify that. Fitting with this: the second stage CAN be mixed (5j), because it decides locally on physical features or centered patches; the U-Net has to model the whole appearance of a cohort.

### 5o Cross-check "truly fixed epoch count" (user approval 21 Sep; queued)
**Starting point (section 11):** the schedule runs a fixed 60 epochs, but the best intermediate checkpoint of the inner fold is used (~11 cases, every 5 epochs) -- e.g. 40 / 60 / 50 / 35 / 60 (seed 42) and 50 / 55 / 10 / 40 / 40 (seed 1) were chosen. **Hypothesis:** this choice is itself a source of noise; with the LAST intermediate checkpoint (learning rate near zero at the end of the cosine schedule, weights "cooled down") the spread between seeds decreases, and the mean F1 stays the same or increases.
**Change:** `cv5.LETZTER_STAND` (default False, behaviour otherwise unchanged; backup cv5.py.vor-letzter-stand-0921) and `turnier.py --letzter-stand` (folder suffix -ls); threshold and minimum size continue to come from the inner fold, for the evaluation the fixed cell counts anyway. **Check (CPU, 2 epochs, inner F1 artificially decreasing):** epoch 2 is chosen (the last checkpoint); the old rule would have taken epoch 1. **Experiment:** base recipe with `--letzter-stand` for seed 42 and seed 1. Since training is deterministic, these runs differ from the existing ones ONLY in the intermediate checkpoint used. **Evaluation (pre-specified):** fixed cell with and without the CSF rule, paired against the respective run with checkpoint selection; the main measure is the GAP between the two seeds (so far 0.585 vs 0.553 raw, 0.646 vs 0.590 with the rule). If the gap shrinks and the mean is not worse, the last checkpoint becomes the recipe.
**Result seed 42 (21 Sep 21:25, 5/5 folds; `ergebnisse/analysis/letzter_stand_s42_valdo*.json`, `operating_point_valdo_letzter_stand_s42.json`):** determinism check passed -- folds 1 and 4, in which the base had chosen epoch 60 anyway, are identical (0.500 / 0.577). Fixed cell raw: **0.538 vs 0.585, paired -0.047 [-0.088; -0.010] (significantly worse)**; P 0.44 vs 0.52, S 0.69 vs 0.67, false positives per case 2.14 vs 1.51. Per cohort -0.050 / -0.025 / -0.074 (all three negative, individually not significant). With the CSF rule at the fixed cell 0.594 vs 0.646. **Is it only due to the threshold?** No: the last checkpoint is more self-confident, its best spot lies at threshold 0.6 instead of 0.3 -- but even there it only reaches 0.618 (optimistically chosen) vs 0.646; cross-selected 0.579 vs 0.622, paired -0.043 [-0.096; +0.011]. **Interpreted:** the checkpoint selection on the inner fold is here NOT merely a source of noise, but acts like early stopping against overfitting (35 training cases): late epochs make the network more sensitive and more self-confident, false positives rise. Notably: with the last checkpoint, the "lucky case" seed 42 (0.594) lands exactly at seed 1 with checkpoint selection (0.590) -- part of its lead is therefore due to well-picked intermediate checkpoints. **Decision:** under the pre-specified rule this is only decided once seed 1 is in (running, ~23:05); for seed 42 the condition "mean not worse" is already violated.
**Result seed 1 and decision (21 Sep 23:04; `ergebnisse/analysis/letzter_stand_s1_valdo_gepoolt.json`, `operating_point_valdo_letzter_stand_s1.json`, `operating_point_valdo_ens2_letzter_stand.json`, `ens2_letzter_stand_valdo.json`):**
| fixed cell 0.3 / 2 mm3 | checkpoint selection s42 | s1 | gap | mean | last checkpoint s42 | s1 | gap | mean |
|---|---|---|---|---|---|---|---|---|
| raw | 0.585 | 0.553 | 0.032 | 0.569 | 0.538 | 0.545 | 0.007 | 0.542 |
| with CSF rule | 0.646 | 0.590 | 0.056 | 0.618 | 0.594 | 0.614 | 0.020 | 0.604 |
Paired per seed: s42 -0.047 [-0.088; -0.010] (raw), s1 -0.007 [-0.068; +0.051] (raw); with the CSF rule and cross-selected threshold s42 -0.043 [-0.096; +0.011], s1 **+0.065 [+0.012; +0.126]** -- the signs are OPPOSITE.
In both seeds the last checkpoint is more sensitive and less precise (s1: S 0.71 vs 0.60, P 0.44 vs 0.51, 2.19 vs 1.42 false positives per case), its best spot lies at threshold 0.5-0.6 instead of 0.3.
Two-seed ensemble (the VALDO standard recipe): last checkpoint 0.571 raw / 0.626 with rule vs 0.594 / 0.647; paired raw -0.023 [-0.069; +0.012] (computed after the fact, not pre-specified).
**Decision under the pre-specified rule:** the GAP between the seeds shrinks as predicted (0.032 -> 0.007 raw, 0.056 -> 0.020 with rule) -- the checkpoint selection on ~11 cases IS a source of noise between seeds; but the MEAN drops (0.569 -> 0.542 raw, 0.618 -> 0.604 with rule), and the ensemble does not improve. Condition 2 violated -> **checkpoint selection stays in the recipe.**
**What follows from this for reporting (more important than the recipe question):** (1) the 0.646 from seed 42 is partly a lottery win in checkpoint selection -- with the last checkpoint the same run drops to 0.594, seed 1 rises from 0.590 to 0.614. Both procedures deliver a mean of ~0.60-0.62; this matches the mean of 0.609 over six nearly identical runs (section 3). In the paper the MEAN over seeds with range therefore belongs in the main table, not the best run.
(2) The fixed cell (threshold 0.3) was chosen on models with checkpoint selection and disadvantages late, self-confident checkpoints; anyone who changes the recipe must also check the operating point. (3) For the question "120 epochs in general?" (5h): longer training only helps here TOGETHER WITH the checkpoint selection. **Limits:** two seeds; the mean difference (-0.014 with rule) lies below the noise floor of 0.03 -- "worse" is not significant for the mean, "not better" is.

### 5p Augmentation, cleanly separated (user question 21 Sep evening; queued)
**Status:** the recipe only mirrors (three axes) and places the bleed randomly in the patch. The only test so far (4.7) was a PACKAGE of z-normalisation and augmentation (rotation, scale, contrast, noise) on the old, inpainted grid: -0.060, not separable. Augmentation ALONE has therefore never been measured. **What exists and what fits our problem:** (a) spatial (rotation, scale, elastic) -- interpolates the few voxels of a bleed; our own finding 4.8 (conversion costs 15-30 % contrast) and the warning in the MicrobleedNet paper argue against it; (b) intensity ONLY (gamma, smooth field, contrast, noise) -- nothing gets interpolated; targets scale and scanner differences, so more aimed at transfer between cohorts (5n) than at CV within a cohort; (c) THICK-SLICE SIMULATION -- thin-slice cases are averaged in z and back-converted, the label follows along; targets exactly our weakest data (VALDO-3 F1 0.479 and VALDO-1 0.557 vs 0.634 for thin slices; catalina-T2* at 5 mm); (d) synthetic bleeds and mimics (CenSynCMB +2.6 points) -- laborious, deferred; (e) nnU-Net standard set (everything together, there with 1000 epochs) -- augmentation and long training duration belong together (5h). **Change (`code/turnier.py`, backup .vor-aug2-0921):** `--aug-intensitaet` (-augi) and `--aug-schicht` (-augs), each switchable INDIVIDUALLY.
**Check (CPU, synthetic single-slice bleed):** intensity -- label and FRST channel bit-identical, background stays 0, 38 of 40 patches changed; thick-slice -- a 1-slice bleed becomes 2-5 slices (= simulated k), the profile through the bleed 0.95 -> 0.62 / 0.71 / 0.70 / 0.60 (partial volume like real thick slices), background stays 0. **Experiment:** base recipe + `--aug-schicht` on VALDO (queued); evaluation paired at the fixed cell AND per cohort (`evaluate.py --split-valdo`) -- the hypothesis predicts a gain in cohorts 1 and 3, not in cohort 2.
**Second experiment (user approval 21 Sep 20:38, queued as [160]):** base recipe + `--aug-intensitaet` on VALDO -> `ergebnisse/turnier/a03-aniso-e60-gd-augi`. **Pre-specified, before any result is in:**
(1) WITHIN VALDO, paired at the fixed cell against the base (seed 42): the expectation is a tie, NO gain -- the CV cases come from the same three scanners as the training. The change counts as harmless if the paired interval contains zero; for context: two seeds of the base lie 0.032 apart (0.585 vs 0.553), a difference of this size is not interpretable without a second seed.
(2) TRANSFER without adaptation (zero-shot, `cmb.transfer.zero_shot`, mean of the five fold models) to catalina-T2* and catalina-SWI, paired over cases against the existing zero-shot of the base (T2* 0.622, SWI 0.305). HERE the hypothesis predicts a gain, specifically via FEWER FALSE POSITIVES per case (5n: transfer costs via false positives, not via annotation style; SWI fails on contrast and scale). Confirmed if the paired interval lies above zero in at least one cohort and not significantly below it in the other; otherwise refuted. The same two zero-shot runs get `-augs` (thick-slice; catalina-T2* has 5-mm slices) -- queued [161]-[164].
(3) NOISE LEVEL: the zero-shot value also depends on the seed of the training; for this the base is likewise transferred without adaptation with seed 1 ([165], [166]). An augmentation gain only counts if it is larger than the gap between the two base seeds. **Consequence:** confirmed -> augmentation goes into the recipe for transfer runs, next test is fine-tuning from the `-augi` start; refuted -> dropped, negative finding stands.
**Result thick-slice simulation within VALDO (22 Sep 13:34; chosen intermediate checkpoints 55 / 50 / 55 / 55 / 45; `ergebnisse/analysis/valdo_augs_*.json`, `operating_point_valdo_augs.json`): the hypothesis is REFUTED.**
Fixed cell raw **0.458 vs 0.585, paired -0.127 [-0.179; -0.075]** (above the noise level of 0.08 from section 3); P 0.34 vs 0.52, S 0.70 vs 0.67, false positives per case 3.3 vs 1.5. With CSF rule 0.522 vs 0.646; with its own, cross-selected threshold (0.6-0.7 instead of 0.3) 0.573 vs 0.622, -0.050 [-0.115; +0.024]. Per cohort: 4 mm -0.045 (n = 8, n.s.), 0.8 mm -0.125 [-0.198; -0.063], **3-4 mm -0.146 [-0.284; -0.012]** -- the hypothesis predicted a GAIN for the thick cohorts; what occurred there was the largest loss (VALDO-3: 3.5 false positives per case, P 0.21). **Interpreted:** the network, which learns artificially smeared bleeds with a stretched label in 35 % of the patches, becomes more self-confident and less precise -- it learns "faint and extended is also a bleed," and that is exactly what the mimics in the thick cohorts are (vessel cross-sections, sulcal partial volume). ~~The simulation produces images that resemble the thick cohorts, but labels that do not occur there (the reference rater marks small, not extended, on the thick slice).~~
**CORRECTED on 22 Sep 21:24 -- this explanation was wrong and is refuted by re-measurement.** Extent of the reference bleeds in z on the common grid (median [10th-90th percentile], voxel size 1 mm): cohort 1 (4 mm) **4.0 [4; 8]**, cohort 2 (0.8 mm) 3.0 [2; 4], cohort 3 (3-4 mm) **4.0 [3; 7]** -- real thick-slice annotations ARE extended in z on the grid, and indeed exactly on the order of magnitude that the simulation produces (k = 2..5 stretches 3 voxels to 4-8).
So the label handling of the augmentation was correct; my explanation was a plausibility story that I had not measured -- the same mistake as the one it was supposed to explain.
**The actual explanation is provided by the robustness probe (5v):** the network is extremely sensitive to z-smearing (sensitivity 0.66 -> 0.43 -> 0.25 -> 0.17 -> 0.13 for k = 2..5, with INCREASING precision 0.63 -> 0.86). Bleeds smeared in 35 % of the training patches thus force the network to accept weaker and more diffuse evidence in order to find them at all. Precisely this lowered bar then fires on every diffuse structure on sharp test images: S 0.70 (higher than the base), P 0.34, 3.3 false positives per case. This fits with the fact that the run at ITS OWN operating point (threshold 0.6-0.7 instead of 0.3) is only -0.050 [-0.115; +0.024] behind: a large part of the deficit is calibration, not capability. **The lesson stands, but phrased differently:** an augmentation that makes the task harder during training shifts the network's operating point -- whoever uses it must also determine the operating point (or re-choose it per model) and expect more false positives on sharp data. **Decision:** discarded within VALDO. The zero-shot runs on catalina stay queued (there the main measure is pre-specified) -- expectation after this finding: more false positives there too, no gain.
**Result intensity augmentation within VALDO (22 Sep 15:18; chosen intermediate checkpoints 35 / 60 / 20 / 55 / 35; `ergebnisse/analysis/valdo_augi_*.json`, `operating_point_valdo_augi.json`): a tie was expected, what is measured is a loss above the noise level.** Fixed cell raw **0.477 vs 0.585, paired -0.108 [-0.180; -0.030]**; with CSF rule 0.527 vs 0.646; cross-selected 0.458 vs 0.622. Per cohort -0.139 [-0.246; -0.065] / -0.113 / -0.058. Unlike the thick-slice simulation, BOTH axes drop here: S 0.56 vs 0.67 AND P 0.41 vs 0.52 (1.9 false positives per case) -- with the same budget the network has learned less, not more wrong things. The run lies below the weakest of the three seeds (0.501). **Interpreted:** five operations with p = 0.5 each hit a patch 2.5-fold on average; gamma on the inverted scale systematically changes the lesion contrast (g < 1 lifts the background, g > 1 pushes it down); combined, the perturbation is apparently too strong for 60 epochs and 35 training cases -- the network normalises per instance anyway and gains nothing from the spread, but pays with learning budget. Whether the base is already invariant to such perturbations (in which case training with them can only cost) is measured by the robustness probe (5v) tonight. **Status of the augmentation question:** both variants are significantly worse within VALDO (thick-slice -0.127, intensity -0.108); the main measure for transfer (zero-shot on catalina, 5p (2)) runs next -- after S 0.56 within the domain I expect no gain there.
**Main measure: transfer without adaptation to catalina (22 Sep 16:37; `mb-arena/transfer/auswertung_zeroshot_t2star_aug_0922.json`, `..._swi_aug_0922.json`; sums):**
| Zero-shot (mean of the 5 fold models) | catalina-T2* F1 (FP per case) | paired against base s42 | catalina-SWI F1 (FP per case) | paired |
|---|---|---|---|---|
| Base s42 | 0.622 (4.8) | -- | 0.305 (6.7) | -- |
| Base s1 = noise level | 0.638 (4.0) | +0.016 [+0.003; +0.033] | 0.311 (5.4) | +0.006 [-0.015; +0.030] |
| Intensity augmentation | 0.608 (5.9) | -0.014 [-0.034; +0.000] | 0.297 (9.0) | -0.007 [-0.029; +0.013] |
| Thick-slice simulation | 0.606 (5.5) | -0.016 [-0.037; +0.002] | 0.240 (16.3) | **-0.065 [-0.096; -0.036]** |
**Decision under the rule from (2) and (3): both hypotheses REFUTED.** No interval lies above zero; the predicted mechanism (fewer false positives per case) occurs nowhere -- both augmentations INCREASE the false positives in both cohorts (T2* 4.8 -> 5.5-5.9; SWI 6.7 -> 9.0 and 16.3 respectively), the thick-slice simulation on SWI significantly and massively so. The noise level of the transfer is small at +0.016 / +0.006; both augmentations lie in the same order of magnitude with the opposite sign. **Side finding that matters for delivery:** seed 1 is the weaker one within VALDO (0.553 vs 0.585), but transfers BETTER (0.638 vs 0.622, significant; 4.0 instead of 4.8 false positives) -- the CV ranking of the seeds does not predict transfer. This supports the two-seed ensemble as the delivered model and warns against choosing "the best seed." **5p balance:** three pre-specified measurements, three negative findings (VALDO -0.127 / -0.108, transfer 0 to -0.065). Augmentation in this form is dropped; whether a perturbation hits a real weakness of the network at all is answered by the robustness probe (5v) BEFORE any further attempt.
(Numbers at the time of queuing; each 2 higher since the second stage was moved forward on 21 Sep 21:52.) **Limits:** one seed per variant; the strengths (gamma 0.7-1.5, field exp(N(0, 0.15)), contrast 0.85-1.15) are set, not tuned -- a null finding refutes this setting, not the idea.

### 5q Interim balance: what has been learned and which search space remains (user question 21 Sep 21:00)
**Learned (each with evidence):**
(1) DATA BEATS METHOD. All large gains lay in the data pipeline: removing original inpainting +0.136 significant (4.1), keeping FRST +0.053 (4.11), z = 1 mm +0.036 (4.8), CSF rule +0.02 to +0.06 (4.3), fixed operating point (5j).
(2) NO TRAINING LEVER RAISES THE F1: blob -0.044, center map -0.054, fnrw -0.030, hard negatives +-0, ratio 1:2 collapses, tissue channels -0.058, own inpainting without gain (5g, 5j-5m). The pattern is always the same: more sensitive (small bleeds 0.52 -> 0.64 / 0.66), paid for with false positives. The bottleneck is not finding, but SEPARATING bleed from mimic with only one sequence. This fits: in CenSynCMB exactly these two levers helped (FN reweighting +1.8, center map +4.5 points) -- there with T1 + T2 + T2*. Sensitivity only pays off if there is something that can separate the additional candidates.
(3) NOISE FLOOR ~0.03: six nearly identical runs 0.590-0.646 (section 3); a gain by the random-forest classifier dissolved with an independent first stage (+0.033 -> +0.004), that of the 3D-CNN was confirmed (+0.044 / +0.045). Without a second seed or an independent first stage, nothing below 0.03 is interpretable.
(4) MIXING HELPS AT THE CANDIDATE LEVEL, NOT AT THE VOXEL LEVEL: U-Nets per cohort; mixed or pre-trained + fine-tuned worse (-0.02 to -0.045, via false positives, 5h, 5n); ONE shared classifier over all cohorts: SWI +0.045, reproduced (5j; the user's idea).
(5) FEW CASES CARRY THE POOLED NUMBERS: three catalina-T2* cases = 28 % of the bleeds and roughly half of those missed (5i); without them F1 0.696.
(6) My expectations were wrong several times (mixing, center map, fine-tuning from the mixed network, "cerebellum gap") -> pre-specify, measure paired, count cases.

**Search space -- exhausted (do not touch again without a new reason):** loss functions and instance-aware training, sampling ratio, hard negatives, tissue as network input, inpainting at every dose, mixing strategies, pre-training + fine-tuning, second stage PER COHORT (logistic, random forest, boosting, CNN), ensembles beyond two members, mirror TTA on catalina.
**Running (queue; after this the training recipe is fully measured):** last intermediate checkpoint (5o), 120 epochs (5h), patch size, augmentation individually (5p), third seed.
**Open, ordered by expected yield per effort:**
(a) MEASURE LABEL CEILING (no GPU; the user's expert judgement on the review page 5b, 201 candidates): if some of the "false positives" are real bleeds, precision is higher than measured, and it becomes visible how close 0.65 is to what the reference actually gives. The VALDO paper reports no inter-rater agreement (section 6). Without this number we may be optimising against label noise.
(b) EXTEND THE SECOND STAGE (the only direction with a reproduced gain; cheap, 216,081 parameters): larger or more elongated context, or MinIP views (mimics are mostly vessels and are recognisable by their continuation across slices); candidates from several first-stage seeds as training material; disagreement between seeds or TTA spread as a feature; pre-trained 3D encoder or a real ResNet; confirmation on VALDO / T2* with a second seed (still pending).
(c) SYNTHETIC BLEEDS AND MIMICS (CenSynCMB: +1.4 and +1.2 points on exactly this dataset; the only literature lever we have not measured). For the SECOND stage much cheaper than for the first: it only needs patches, not whole volumes, and it hits exactly the bottleneck (more hard negatives and positives for the classifier). Effort several days, no code published.
(d) RE-MEASURE ARCHITECTURE ON CLEAN DATA: the tournament (4.4) ran with 40 epochs and early stopping on the OLD, inpainted grid -- the ranking (aniso wins, large networks "not trainable") comes from damaged data and a short budget. Expected gain small, but a reviewer question; 2-3 networks x 120 ep. ~ 10-15 GPU hours.
(e) UPPER BOUND OF THE SINGLE-SEQUENCE SOLUTION (VALDO only): T1 + T2 + T2* as input. Not usable for catalina (no T1, FLAIR only a subgroup, 4.10), but delivers the number of how much of the gap to CenSynCMB's 74.3 is due to the sequences.
(f) SMALLER ITEMS: percentile scale p99.5 (+0.025 n.s., the only positive sign in preprocessing, second seed dropped on 21 Sep, 4.9); training at native resolution instead of conversion (4.8: conversion costs 15-30 % contrast; laborious, outcome open); threshold per cohort. NOT AVAILABLE: phase image to separate calcium / blood (VALDO does not provide one; to be asked of the user for catalina-SWI).
**Recommendation:** (a) first, because it puts every further number into context; then (b) and (c) together -- synthetic mimics only for the classifier; (d) in GPU gaps. None of this has been started; this list is a plan, not a result.

### 5r Extending the second stage: more context (5q b; user approval 21 Sep evening; queued)
**Original:** MicrobleedNet checks candidates with a second network on 24^3 patches. **Our status:** the shared small 3D-CNN (5j) looks at a cube of 16 mm around each candidate (16 x 32 x 32 voxels at 1 x 0.5 x 0.5 mm); 20 mm was what was stored. **Observation:** the mimics are mostly vessels, and a vessel gives itself away by its CONTINUATION -- through the slices or in-plane. At 4-5 mm slice thickness (VALDO-1 / -3, catalina-T2*) the 16-mm cube contains only 3-4 measured slices. **Hypothesis:** a second, coarse view (28 mm at half resolution) lowers the false positives at the same sensitivity, most strongly in the cohorts with thick slices.
**Change:** `candidates.py --half 32 32 16` stores 32-mm cubes (default unchanged at 20 mm); `classifier.fit_predict_cnn_two_scale` = the same encoder TWICE (fine: central 16 mm at full resolution; coarse: 28 mm, pre-averaged 2x), the two 64-dimensional feature vectors are concatenated before the linear layer; loss, optimiser, 40 epochs, three members, mirroring and offset as in the fine CNN, so the comparison measures only the context; `pooled.py --models cnn cnn2s`. Backups `*.vor-zweiskalig-0921`. **Check (CPU):** (1) old patch size: old and new version of the fine CNN deliver bit-identical values; (2) on the large patches the fine CNN delivers exactly the same as on their central 20 x 40 x 40 -- so the previous numbers remain reproducible; (3) if only the margin OUTSIDE the fine view is changed, the two-scale network reacts, the fine one does not; (4) patches that are too small are rejected. **Experiment:** ONE run computes `pooled cnn` and `pooled cnn2s` on the same candidates (VALDO, T2*, SWI; candidates as before, only with a larger patch). **Pre-specified:** the main measure is the paired difference cnn2s - cnn per cohort (bootstrap over cases; `cmb.stage2.compare_runs`). cnn2s is adopted only if the difference lies significantly above zero in at least one cohort and significantly below zero in none, AND if this repeats with the independent SWI stage (seed 1) -- the same hurdle that the fine CNN cleared and the random-forest classifier did not (5j). **Built-in control:** `pooled cnn` must hit the previous values (0.625-0.636 / 0.672 / 0.670; the GPU does not compute bit-identically, two earlier runs were 0.005 apart). **Limit:** one second-stage seed per run (three averaged members); we will only know the reasons for the false positives once the expert judgement (5b) is in -- if it says "mostly vessels," that supports the approach; if it says "mostly real bleeds," the second stage is the wrong tool.
**Result main run (22 Sep 00:08; GPU 67 min; `mb-arena/stage2/pooled_h32_cnn_cnn2s_0921.json`):** built-in control passed -- `pooled cnn` on the large patches hits the old values (0.626 / 0.672 / 0.670 vs 0.625-0.630 / 0.672 / 0.670).
| | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| threshold only | 0.565 | 0.654 | 0.626 |
| fine CNN (16 mm) | 0.626 (P 0.63, S 0.63, 0.91 FP per case) | 0.672 (P 0.70, S 0.65, 3.04) | 0.670 (P 0.74, S 0.61, 2.26) |
| two-scale (16 + 28 mm) | 0.629 (P 0.66, S 0.60, 0.77) | **0.687** (P 0.73, S 0.65, 2.71) | **0.686** (P 0.74, S 0.64, 2.46) |
| **paired two-scale - fine (main measure)** | +0.003 [-0.026; +0.036] | +0.016 [-0.002; +0.030] | +0.016 [-0.002; +0.037] |
| paired two-scale - threshold only | +0.064 [+0.017; +0.105] | +0.033 [+0.017; +0.055] | +0.060 [+0.035; +0.094] |
Per VALDO cohort (two-scale - fine): 4-mm slices (n = 8) -0.028 [-0.110; +0.000]; 0.8 mm (n = 27) -0.008 [-0.054; +0.039]; 3-4 mm (n = 22) **+0.057 [+0.000; +0.107]** (false positives 19 -> 13 at the same 17 hits).
**Read against the pre-specified rule:** the main measure is significant in NO cohort (twice +0.016 with a lower bound of -0.002) -> condition 1 is narrowly NOT met; nothing is adopted for now. The direction fits the hypothesis where it predicts it: on T2* (5-mm slices) and VALDO-3 the false positives drop at the same sensitivity; on SWI, by contrast, the gain comes via sensitivity (a different mechanism than predicted), on VALDO-1 (n = 8) it points in the wrong direction. Against the pure threshold, the two-scale stage is for the first time significantly better in ALL three cohorts (the fine CNN was not, on VALDO). The confirmation run with the independent SWI stage is running; only that will decide.
**Confirmation run and decision (22 Sep 01:16; `mb-arena/stage2/pooled_h32_cnn_cnn2s_swi_s1_0921.json`; SWI candidates from the independently trained first stage, VALDO and T2* candidates unchanged):**
| | VALDO | catalina-T2* | catalina-SWI (first stage seed 1) |
|---|---|---|---|
| fine CNN | 0.632 (P 0.66, S 0.60, 0.75 FP per case) | 0.672 (P 0.71, S 0.64, 2.87) | 0.665 (P 0.73, S 0.61, 2.48) |
| two-scale | 0.601 (P 0.61, S 0.60, 0.95) | 0.691 (P 0.74, S 0.64, 2.43) | 0.678 (P 0.75, S 0.62, 2.26) |
| **paired two-scale - fine** | **-0.030 [-0.059; -0.011]** | **+0.019 [+0.001; +0.036]** | +0.013 [-0.001; +0.031] |
**Decision under the pre-specified rule: NOT adopted.** In the main run no cohort was significant, in the confirmation run T2* is significantly better, but VALDO is significantly WORSE -- the condition "significantly below in no cohort" is violated. The final chain keeps the fine CNN. **What remains:** (1) on catalina-T2* (5-mm slices) the picture is the same in both runs -- +0.016 / +0.019, false positives 3.04 -> 2.71 and 2.87 -> 2.43 respectively at the same sensitivity --, exactly the predicted mechanism in the cohort with the thickest slices; on SWI positive twice, not significant twice (+0.016 / +0.013).
(2) **The second stage has its own training noise that our intervals do NOT capture:** between the two runs the VALDO candidates are identical, only the SWI part of the training pool has changed -- yet the two-scale network on VALDO drops from 0.629 to 0.601, the fine one goes from 0.626 to 0.632. The bootstrap over cases measures the uncertainty of the CASE SELECTION for ONE trained classifier, not that of the training; a "significant" -0.030 is therefore less certain than the interval suggests (the same applies retroactively to the +0.044 / +0.045 of the fine CNN on SWI -- there, however, it came out the same twice with an independent first stage). The clean approach would be: train each variant of the second stage 5x with different seeds and report the spread as well (~6 GPU hours per comparison).
(3) An afterthought, NOT a decision: using the coarse view only for thick-slice cohorts would contradict the idea of ONE classifier and would be chosen by looking at the data. **Recommendation:** no further GPU effort for the context; the expected gain (<= 0.02) lies below the noise. More promising remain the label ceiling (5b) and synthetic mimics (5q c).
**Not yet started (5q b, c):** candidates from several seeds as training material (for T2* the second seed is missing), disagreement between seeds as a feature, pre-trained encoder, synthetic bleeds and mimics for the classifier (design follows once 5b and this run show WHICH mimics are missing).

### 5t Patch size as a dose series (user question 21 Sep early on Sundaresan's patch system; small patch done 22 Sep 07:43)
**Original:** MicrobleedNet trains the candidate stage on 48^3-voxel cubes of the original grid (~24-48 mm for thin slices). **Ours:** 38 x 62 x 94 voxels on the 0.5 x 0.5 x 1 mm grid = 38 x 31 x 47 mm, 800 per epoch, sliding window in prediction. **Question:** does Sundaresan's small patch help (more patches per case, less background per patch), or does the missing context cost? **Series:** small 24 x 48 x 48 voxels (24 x 24 x 24 mm) -- recipe -- large 64 x 128 x 128 (64 x 64 x 64 mm, queued); otherwise everything unchanged, including the 800 patches per epoch.
**Result small (`ergebnisse/analysis/valdo_patch24_*.json`, `operating_point_valdo_patch24.json`; ~10 instead of ~21 min per fold):** fixed cell raw **0.369 vs 0.585, paired -0.216 [-0.277; -0.156]**; P 0.25 vs 0.52, S 0.74 vs 0.67, false positives per case 5.5 vs 1.5. With CSF rule 0.491 vs 0.646; even with its own, cross-selected threshold (0.4-0.6 instead of 0.3) only 0.523 vs 0.622, paired -0.099 [-0.170; -0.040]. Per cohort -0.080 (4 mm, n.s.) / -0.186 (0.8 mm) / -0.267 (3-4 mm), both significant; worst in the cohort with the largest voxels and the most non-brain in the mask (VALDO-3: 6.8 false positives per case).
**Interpreted:** the small patch finds more (S 0.74) and separates much worse -- the network sees 24 mm of context and can no longer distinguish vessels, sulci, and brain edge from bleeds; the mimics are precisely the structures that only give themselves away over a few centimetres. This is the same pattern as all the sensitivity levers (5j-5m), only stronger. **Two confounds, honestly named:** (1) with 800 patches per epoch, the network sees only a quarter of the voxels per epoch (55k vs 221k per patch) -- less context AND less training data per epoch; per 5h, however, the budget effect is small against -0.216. (2) Sundaresan's recipe compensates for the small patch through the SECOND stage (24^3 classifier); here we have only measured the first stage. The statement is therefore: for a SINGLE-STAGE network, a 24-mm patch is significantly too small. **Decision:** small discarded; the recipe stays until the large patch is measured (rule pre-specified: large is only adopted if paired significantly better than the recipe AND not worse in any cohort; at most a small gain is expected, because the tournament winner already uses the anisotropic first stage and the context limit lies more with the mimics than with the network).
**Large patch 64 mm (started 22 Sep 07:39): not measurable in this setup.** The buffer per epoch (800 patches x 3 channels x 1.05M voxels x 4 B, held twice in memory) lifts the process to 33-35 GiB and thus above the intentional memory limit of the service cgroup (MemoryHigh 32 GiB): throttling, 0 % GPU. It is not the limit that gets changed, but the dose: substitute point **48 x 96 x 96 voxels = 48-mm cube** (2x the voxels of the base, ~24 GiB), queued. The series therefore reads 24 mm -- recipe (38 x 31 x 47 mm) -- 48 mm; 64 mm would need half as many patches per epoch (code change) and ~8 GPU hours.
**Result 48 mm (22 Sep 10:17; chosen intermediate checkpoints 60 / 45 / 20 / 45 / 45; ~29 min per fold; `ergebnisse/analysis/valdo_patch48_*.json`, `operating_point_valdo_patch48.json`):** fixed cell raw 0.569 vs 0.585, paired -0.016 [-0.074; +0.034]; P 0.49 vs 0.52, S 0.67 vs 0.67, false positives per case 1.67 vs 1.51. With CSF rule 0.609 vs 0.646, cross-selected 0.610 vs 0.622 (-0.012 [-0.084; +0.050]). Per cohort -0.018 / -0.013 / -0.028, all n.s.
**Dose series complete:** 24 mm 0.369 -- recipe 38 x 31 x 47 mm **0.585** -- 48 mm 0.569 (raw); with CSF rule 0.491 -- **0.646** -- 0.609. **Decision under the pre-specified rule: 48 mm not adopted** (not significantly better; not significantly worse in any cohort -- the difference lies within the noise floor). **Interpreted:** downward, patch size is a sharp lever (-0.216), upward a blunt one (-0.016 n.s.): from about 3-4 cm of context onward, the network has what it needs for separation; more context brings nothing at 800 patches per epoch, but costs double the compute time (29 vs 21 min per fold) and memory. Sundaresan's 48^3-voxel cubes are, for thin slices, physically closer to our 24-mm point than to our recipe; his recipe carries the small patch through the second stage. **Limits:** one seed per point (noise floor 0.03; the 48-mm difference lies below it); the 800 patches per epoch are fixed, so the large patch sees twice as many voxels per epoch -- an advantage that did not help it; 64 mm remains unmeasured (memory, see above).
### 5v Which augmentation fits CMB -- a procedure instead of plausibility (user question 22 Sep 13:30; tool built, GPU run queued)
**Trigger:** 5p -- the thick-slice simulation was plausibly justified, cost 1.7 GPU hours, and caused confirmed harm (-0.127), most strongly where the gain had been predicted. The error was methodological: image statistics of the target cohort were imitated, label statistics were not; and it was only measured after training in the first place. **Procedure (five steps, each before the next training run):**
(1) *Write down invariances.* What is allowed to change without changing the label: orientation without interpolation (mirroring, 90-degree rotation), global brightness, bias field, noise, contrast. What is not: size in mm, roundness, low-contrast detail, the relationship of the slices to each other. Everything from the second list (rotation with interpolation, scale, thick-slice) is suspect until it is measured.
(2) *Robustness probe on the finished network, without training* (`cmb/analysis/robustness_probe.py`, new): the saved fold models predict their own test cases after the IMAGES have been perturbed in memory -- seven perturbations at 22 strengths (gamma 0.7-1.5, bias field exp(N(0, 0.05-0.3)), contrast x0.7-1.3, noise 2-10% of the brain SD, thick-slice k = 2-5, rotation 5 / 15 degrees, scale 0.9 / 1.1; geometric perturbations are applied to the image and the FRST, the probability map is rotated back, the label stays untouched). Scored at the fixed operating point with the CSF rule, paired against the unperturbed prediction of the SAME run, everything out-of-fold. Reading rule, fixed in advance: if the F1 drops markedly at a realistic strength (the strength the target cohort actually shows) -> the network is not invariant, training with this perturbation is a candidate; if it does not drop -> training with it brings nothing within the domain; if it already drops at mild, unrealistic strengths -> the perturbation destroys the signal, do not train with it. Cost: prediction instead of training, ~2 GPU hours for all 22 conditions on 57 cases (CPU smoke test on one case passed, 11 conditions, unperturbed chain = baseline value). Queued as [172] at the end of the queue.
(3) *Measure realism against the target cohort* (`cohort_differences.py`, existing): an augmentation fits if it shifts the VALDO statistics toward catalina -- e.g. catalina bleeds are paler (relative contrast 0.20-0.21 vs. 0.34-0.38). An intensity augmentation that LOWERS the lesion contrast would be the right fit; the queued one (`--aug-intensitaet`) only scatters around the starting value. This is checked against the statistics after the zero-shot result (5p).
(4) *Label check:* after augmentation, the label must look like real labels from the target domain (size in voxels, extent in z). This is exactly where the thick-slice simulation failed (label 2-5 slices; real thick-slice labels are 1 slice).
(5) *Only then train*, only candidates that pass (2)-(4); fix the expected win location in advance; second seed for anything below 0.08 (Section 3); read the result by mechanism (P / S / false positives), not just by F1 -- for CMB, any augmentation that raises sensitivity and lowers precision is a false-positive generator.
**Expectation for the probe (in advance):** gamma / contrast / noise at mild strengths without effect (the network normalizes per instance), at strong strengths a drop via false positives; thick-slice clearly from k = 3 on (cohorts 1 / 3 are already weaker); rotation 15 degrees and scale 0.9 / 1.1 clearly (interpolation costs low-contrast detail, 4.8). If this holds, only the photometric augmentation aimed at the target direction (3) remains as a candidate for transfer.
**Result (22 Sep 21:24, 57 cases x 23 conditions, 2.5 GPU-h; `ergebnisse/analysis/robustness_probe_valdo_s42.json`; unperturbed = 0.646, i.e. the reported number -- built-in control passed):**
| Perturbation | Strength | F1 | P | S | FP/case | paired against unperturbed |
|---|---|---|---|---|---|---|
| Gamma | 0.85 / 1.15 | 0.630 / 0.642 | 0.64 / 0.61 | 0.63 / 0.68 | 0.9 / 1.1 | -0.015 / -0.004 (n.s.) |
| Gamma | 1.5 | 0.583 | 0.49 | 0.72 | 1.8 | **-0.063 [-0.117; -0.020]** |
| Bias field | 0.05 / 0.15 / 0.30 | 0.652 / 0.612 / 0.601 | | | | +0.007 / **-0.034 [-0.058; -0.005]** / -0.045 |
| Contrast | 0.7 / 0.85 / 1.15 / 1.3 | 0.608 / 0.626 / 0.643 / 0.628 | | | | **-0.037 [-0.068; -0.006]** / -0.020 / -0.003 / -0.017 |
| Noise | 0.02 / 0.05 / 0.10 | 0.641 / 0.645 / 0.633 | | | | -0.005 / -0.000 / -0.013 (all n.s.) |
| **Thick-slice** | k = 2 / 3 / 4 / 5 | 0.536 / 0.363 / 0.287 / 0.225 | 0.71 / 0.65 / 0.86 / 0.86 | **0.43 / 0.25 / 0.17 / 0.13** | 0.4 / 0.3 / 0.1 / 0.1 | **-0.110 / -0.283 / -0.358 / -0.421** |
| Rotation | 5 / 15 degrees | 0.645 / 0.627 | 0.73 / 0.76 | 0.58 / 0.53 | 0.5 / 0.4 | -0.000 / -0.018 (n.s.) |
| Scale | 0.9 / 1.1 | 0.648 / 0.627 | 0.75 / 0.78 | 0.57 / 0.53 | 0.5 / 0.4 | +0.002 / -0.019 (n.s.) |
**Read against the expectation:** (1) Photometry -- confirmed. Mild changes bounce off (the network normalizes per instance), only gamma 1.5 costs anything, and it does so via false positives (P 0.49). The network is invariant to exactly the perturbations that `--aug-intensitaet` trained with (gamma 0.7-1.5, field 0.15, contrast 0.85-1.15) -- training with it could therefore only cost learning budget, which is what 5p measured. **The probe would have saved this run.**
(2) Thick-slice -- confirmed and stronger than expected: by far the biggest weakness, k = 5 costs 0.42 F1 and two-thirds of the sensitivity. Precision RISES in the process (0.63 -> 0.86): smearing erases bleeds, it does not invent any.
(3) Rotation and scale -- **my expectation was wrong.** The F1 does not move (all four n.s.), but the composition shifts strongly: precision 0.63 -> 0.73-0.78, sensitivity 0.66 -> 0.53-0.58, false positives 0.95 -> 0.37-0.51. Interpolation erases SMALL STRUCTURES INDISCRIMINATELY -- bleeds and mimics alike -- and the two losses cancel out in the F1. This is the measurement from 4.8 (conversion costs 15-30% contrast): it costs roughly an eighth of the bleeds. **Methodologically:** "F1 unchanged" does NOT mean "invariant"; without P / S / FP I would have drawn exactly the wrong conclusion here. The rule "read by mechanism, not by F1" (5v step 5) has thus proven itself for the second time today.
**What follows from this:** (a) Photometric augmentation is done for this network -- it is already invariant. (b) The only real weakness is the z-resolution; it explains the cohort ranking (0.634 thin vs. 0.557 / 0.479 thick) physically rather than methodologically. A naive training attempt against it failed (5p) because it shifts the operating point; a per-cohort operating point or a per-slice-thickness model would be more promising -- neither has been measured yet. (c) Geometric augmentation stays dropped, but for a different reason than thought: not because it confuses the network, but because it makes an eighth of the bleeds invisible.
**Limitations:** one run (seed 42); the thick-slice perturbation hits the already-thick cohorts 1 / 3 a SECOND time -- the values are therefore "additional loss of z-resolution," not "what 4-mm slices cost."

### 5s Re-measuring architecture on clean data (5q d; user approval 21 Sep evening; queued at the end of the queue)
**Observation:** the tournament (4.4) ran with 40 epochs and early stopping on the old, inpainted grid; the inpainting hit 109 of 236 bleeds (4.1). The ranking (aniso 0.404, nnU-Net design 0.265, Attention U-Net 0.245) therefore comes from damaged data and a short budget -- a reviewer will ask whether it holds on clean data. **Experiment:** `a02-nnunet` and `a01-attunet` in today's recipe (`--epochen 60 --gitter dev/gitter_d`, seed 42), paired against a03-aniso-e60-gd at the fixed cell, with and without the CSF rule and per cohort. Deliberately 60 instead of 120 epochs: for 60, the baseline has three seeds as a noise level, and 5o shows that late epochs tend to hurt here.
**Fixed in advance:** the baseline stays unless a challenger is paired-CONFIRMED ahead; if one is within 0.03 (noise floor), it gets a second seed before anything is concluded. The result is reported in any case -- even "ranking holds" is a statement for the paper. **Expectation:** ranking holds, the gap shrinks (the inpainting harmed all networks, the winner no less).
**Runnability checked in advance (21 Sep 22:14, CPU, `turnier.py --probe --gitter dev/gitter_d`: 2 epochs, 7 cases, writes only to `ergebnisse/turnier-probe/`):** a02-nnunet ok (16,544,675 parameters), a01-attunet ok (22,667,381) -- both run in today's scaffold; the shared status file `dev/turnier_probe.json` was afterward reset to the state before the probe (copy of the CPU probe: `dev/turnier_probe.json.cpu-probe-0921`).
**Result a02-nnunet (22 Sep 18:20; chosen checkpoints 50 / 55 / 15 / 55 / 50; 16.5M parameters; `ergebnisse/analysis/valdo_nnunet_*.json`, `operating_point_valdo_nnunet.json`):** fixed cell raw **0.458 vs. 0.585, paired -0.127 [-0.202; -0.048]** (above the noise level 0.08); P 0.38 vs. 0.52, S 0.57 vs. 0.67, false positives per case 2.2 vs. 1.5; with CSF rule 0.515 vs. 0.646; cross-wise 0.495 vs. 0.622 (-0.127 [-0.245; -0.020]). Per cohort -0.129 / -0.134 [-0.257; -0.027] / -0.093 -- behind in all three cohorts, confirmed in the thin-slice one. All five folds lie below the baseline (0.525 / 0.471 / 0.483 / 0.511 / 0.432 vs. 0.645 / 0.500 / 0.629 / 0.556 / 0.577), even below the weakest baseline seed (0.501 pooled).
**Read:** the ranking from the tournament holds on clean data; at 0.127 the gap is smaller than in the tournament (0.404 vs. 0.265 = 0.139), but not substantially so -- the inpainting harmed both networks by roughly the same amount, and the expectation "gap shrinks markedly" is only just met. Unlike with all training levers, BOTH axes drop here (less sensitivity AND less precision): the isotropic network, with pooling in all three directions from the first stage on, loses the information that the anisotropic first stage preserves at 1-mm slice spacing from 3-4-mm slices (4.4, "trust only the measured direction at first"). **Decision per the rule:** baseline stays; no second seed (gap > 0.03). The second challenger (a01-attunet) is running.
**Result a01-attunet (22 Sep 20:27; chosen checkpoints 50 / 55 / 55 / 50 / 40; 22.7M parameters; `ergebnisse/analysis/valdo_attunet_*.json`, `operating_point_valdo_attunet.json`):** fixed cell raw **0.463 vs. 0.585, paired -0.121 [-0.190; -0.061]**; P 0.35 vs. 0.52, S 0.68 vs. 0.67, false positives per case 3.1 vs. 1.5; with CSF rule 0.536 vs. 0.646; cross-wise 0.548 vs. 0.622 (-0.074 [-0.160; +0.010]). Per cohort -0.070 / -0.073 [-0.151; -0.012] / **-0.200 [-0.358; -0.055]** -- the collapse is in the cohort with the largest voxels (P 0.18, 3.5 false positives per case). Four of the five folds below the baseline (exception fold 1: 0.518 vs. 0.500).
**Read:** the Attention U-Net is NOT a weaker finder (S 0.68 = baseline), but a false-positive generator -- the same axis as all sensitivity levers (5j-5m). The gates weight context that no longer exists at 3-4-mm slices after isotropic pooling. **Balance of the counter-check (5s):** both challengers stay confirmed behind the anisotropic baseline on CLEAN data and in today's recipe; the tournament's ranking (4.4) holds, even though it arose on inpainted data with 40 epochs and early stopping. The gaps shrink slightly (tournament: nnU-Net -0.139, Attention -0.159; now -0.127 and -0.121) -- the inpainting harmed all networks similarly. This answers the reviewer question "does the architecture choice hold on clean data?" with yes, without having to repeat the whole tournament (2 runs instead of 10). **Limitations:** one seed per challenger; the remaining eight tournament designs were not re-measured -- but they were even further behind there (0.080-0.294 vs. 0.404), and two out of two re-measured confirm the ordering.
**User question 22 Sep 20:34: "the Attention U-Net has the highest sensitivity -- can the classifier turn that into a better F1?" -- no, and this is decidable without a GPU.**
For a two-stage chain, the F1 of the first stage is the wrong measure; what matters is the UPPER BOUND of the candidate set (how many reference bleeds are touched at all at the low candidate threshold of 0.15). The second stage can discard false positives, but a bleed that is never found is lost for good. Measured (`cmb.stage2.candidates --keep-csf`, CPU, 4 min):
| first stage | candidates | of which false | bleeds reached (of 139) | upper bound S | F1 ceiling with an error-free classifier |
|---|---|---|---|---|---|
| Baseline a03-aniso | 278 | 172 | 103 | 0.74 | 0.851 |
| a01-attunet | 407 | 297 | **104** | 0.75 | **0.856** |
| Lesion loss (blob) | 370 | 252 | 110 | 0.79 | 0.884 |
| a02-nnunet | 328 | 230 | 92 | 0.66 | 0.797 |
The Attention network reaches ONE more bleed and pays with 125 additional false candidates; its higher sensitivity at the operating point 0.3 is therefore not additional information but more broadly scattered probability mass -- the baseline already suspects the same bleeds at 0.15. Even with an error-free second stage, the chain ended at 0.856 vs. 0.851, a tenth of the noise floor. **The counter-test already exists (5j):** the blob network, with 110 bleeds reached, had real headroom (ceiling 0.884) -- the final chain nonetheless got worse (0.594 vs. 0.625; chain S 0.63 instead of 0.60, but P 0.56 instead of 0.65); the classifier passed most of the additional candidates through as false positives. If +7 reachable bleeds bring nothing, +1 brings nothing. **Decision: no GPU run.**
**Rule for the scaffold (third of this kind today, after 5v and the memory limit):** a first stage is only better for a two-stage chain if it raises the UPPER BOUND, not if it is more sensitive at the operating point. The upper bound costs 4 CPU minutes and must be measured BEFORE every training or stage-2 run. Conversely: the only direction that can truly help the chain is a first stage with a higher upper bound AND a comparable candidate count -- none of the variants measured so far achieves that (blob: +7 bleeds, but +92 candidates).

### 5u Third seed of the baseline and three-seed ensemble (22 Sep 11:57; `ergebnisse/analysis/valdo_seeds_s2_gepoolt.json`, `operating_point_valdo_seeds_s2.json`, `valdo_s2_split.json`)
**Purpose:** put the recipe's noise level on a three-point basis and check whether the ensemble grows beyond two members (5j said: flat beyond two). **Result (seed 2, chosen checkpoints 35 / 35 / 45 / 50 / 45):**
| fixed cell | s42 | s1 | s2 | mean (range) | Ensemble 2 (s42 + s1) | Ensemble 3 |
|---|---|---|---|---|---|---|
| raw | 0.585 | 0.553 | 0.501 | 0.546 (0.084) | 0.594 | 0.581 |
| with CSF rule | 0.646 | 0.590 | 0.578 | 0.605 (0.068) | 0.647 | 0.636 |
Paired against s42 (raw): s1 -0.032 [-0.110; +0.042], **s2 -0.083 [-0.135; -0.033]**, Ensemble 2 +0.009, Ensemble 3 -0.004. Seed 2 per cohort -0.106 / -0.066 [-0.110; -0.024] / -0.093; P 0.42, 2.1 false positives per case (s42: 0.52, 1.5).
**Read:** (1) Seed 42 is the best of three, seed 2 the worst -- the gap between two runs of the IDENTICAL recipe is, at -0.083, larger than most of the "effects" in this document and is flagged as confirmed by the case bootstrap: the noise level in Section 3 is updated accordingly. (2) Ensemble: the third member (the weak one) lowers the ensemble slightly (0.647 -> 0.636); ensemble ~ best member, +0.03 above the mean member -- the statement from 5j holds. The two-seed ensemble remains the standard; a third member is only worthwhile if it is not the weakest, and that is not known in advance. (3) Where the seed acts: precision and false positives, not sensitivity (0.63-0.67 for all three) -- the same axis as for all training levers. **Decision:** the baseline is reported in the paper as 0.605 +- 0.034 (three seeds, with CSF rule); the two-seed ensemble (0.647) as the delivered model.
**Limitations:** three seeds are few for a noise level; the statement "range 0.07-0.08" may still grow with more seeds.
**Addendum 23 Sep 06:40 (user directive: 10 seeds; 5 of 10 done, `ergebnisse/analysis/operating_point_valdo_seeds_s0_s4.json`, `valdo_seeds_s3_s4.json`):** with CSF rule at the fixed cell 0.646 / 0.590 / 0.578 / 0.603 / 0.608 -- **mean 0.605, range 0.068**, i.e. exactly the values from the three seeds (0.605 / 0.068). Raw: 0.585 / 0.553 / 0.501 / 0.550 / 0.545; paired against seed 42, the new ones come out at -0.035 [-0.096; +0.024] and -0.039 [-0.096; +0.011]. The picture is thus stable: seed 42 remains the best of five, the mean does not move, and all four other seeds lie below it. The reported number for a model remains 0.605 +- 0.034; the precision carries the spread (0.45-0.52 raw), the sensitivity does not (0.67-0.69).

### 5w The ceiling of the two-stage chain -- and the first measured way to raise it (22 Sep 22:09; `cmb/analysis/candidate_ceiling.py`, new; `ergebnisse/analysis/candidate_ceiling_valdo.json`)
**Starting point (5s):** the second stage can only discard candidates, never invent a missed bleed. The ceiling of the whole chain is therefore set in the candidate step. So far only ONE point had been measured (baseline run s42, threshold 0.15: 103 of 139, F1 ceiling 0.851 with an error-free second stage -- 0.647 is achieved). Two obvious questions had never been asked: does the ceiling depend on the candidate threshold, and do different seeds miss the SAME bleeds? Both can be answered without training and without a GPU, from the stored probability maps.
| first stage | threshold 0.15 | 0.10 | 0.05 | 0.02 |
|---|---|---|---|---|
| s42 | 103 (ceiling 0.851) | 108 | 112 | 113 (0.897) |
| s1 | 99 | 103 | 103 | 103 (0.851) |
| s2 | 103 | 110 | 112 | 113 (0.897) |
| **UNION of the three** | **112** (0.892) | **116** (0.910) | **120** (0.927) | 121 (0.931) |
| Candidates of the union | 885 | 1127 | 1258 | 1316 |
**(1) The threshold 0.15 gives away bleeds.** Going from 0.15 to 0.05, the baseline run gains 9 bleeds (103 -> 112) for 128 additional candidates; after that it is maxed out (0.02 brings one more). The threshold was never chosen with justification -- it comes from the first attempt on 19 Sep.
**(2) Different seeds miss DIFFERENT bleeds.** At threshold 0.05, s42 reaches six bleeds that no other run reaches, s2 five. The union comes to 120 of 139 (ceiling 0.927) against 112 for the best single run. **This is exactly the information our current ensemble throws away:** `cmb.analysis.ensemble` AVERAGES the probability maps, thereby damping what only one member sees (5j: ensemble ~ best member). At the candidate level, the union is the right combining operator, not the mean -- because the second stage is supposed to sort out the false positives anyway.
**What this is worth:** the ceiling of the chain would rise from 0.851 to 0.927; incidentally, the shared classifier would get around 1258 VALDO candidates instead of 278 (106 positive) (the second stage is data-poor on VALDO, 5j).
**What speaks against it, honestly:** measured today (5s), a more sensitive network with a higher ceiling (blob, 110 reached, ceiling 0.884) made the chain WORSE (0.594 vs. 0.625) -- the classifier passed the additional candidates through as false positives. A higher ceiling is necessary, not sufficient. The difference here: the additional candidates come from the same recipe (same error statistics) and the second stage's training set grows fourfold -- neither was the case in the blob attempt.
**Proposed experiment (not queued, waiting on the user; ~40 min GPU, no new training runs needed):** build candidates from the union of the three existing VALDO seeds at threshold 0.05 (`cmb.stage2.candidates` per run, then merge components -- a tool for this is still missing), a shared 3D-CNN on top of it (`pooled.py --models cnn`), paired against today's chain. **Fixed in advance:** adopted only if the paired gain is above zero AND repeats with an independent second cohort -- the same hurdle as for the CNN (5j). Expectation: the chain's sensitivity rises (0.60 -> 0.65-0.70), precision falls; whether the F1 rises depends entirely on whether the classifier can sort the 980 additional false positives. **For catalina** it would need a second seed per cohort (T2* missing, ~1.7 GPU-h each).
**Limitations:** the ceiling is computed with the fixed post-processing (>= 1 mm3, remove blobs); 19 bleeds remain untouched by any run even at threshold 0.02 -- no two-stage procedure is responsible for these, only the first stage or the reference (5b).

### 5x How much of our error is the reference? -- three proxy measurements without a rater (22 Sep 22:20; `cmb/analysis/label_noise_proxies.py`, new; `ergebnisse/analysis/label_noise_proxies_valdo.json`)
**Trigger:** the blinded review page (5b) answers the question cleanly, but needs a clinician. The user clarified on 22 Sep that he is not one -- my planning had silently assumed this since 21 Sep, without ever checking it. Until a rater is available, three objective proxy measurements narrow the question down. Paths checked before but found unsuitable: VALDO delivers only ONE reference mask per case (no rater masks, checked against the source dataset); the 8 possible duplicate persons between catalina-T2* and -SWI ALL carry no bleed, nothing follows from two empty masks.
**Measured on the reported run (baseline run s42, fixed cell with CSF rule: 92 hits, 54 false positives, 47 missed, F1 0.646):**
| | confirmed by SHIVA-CMB | confirmed by both other seeds |
|---|---|---|
| 94 hit components | 66 (70%) | 75 (80%) |
| **54 false positives** | **17 (31%)** | 28 (52%) |
| 47 missed reference bleeds | 10 (21%) | 2 (4%) |
**(1) A third of our false positives is also found by an outside method.** SHIVA-CMB is trained on 450 other scans and does not know our chain. It independently flags 17 of 54 false positives; **16 of those are additionally found by all three of our seeds.** These are 16 locations where four independently trained networks see a bleed and the reference says no. If every false positive confirmed by SHIVA is counted as a true bleed, the F1 would rise from 0.646 to **0.722** -- that is the upper bound of what this proxy justifies, NOT an estimate. For context: SHIVA finds only 70% of the bleeds confirmed by the reference; if all 54 false positives were real, it would have to find around 38 of them, it finds 17.
**(2) The missed bleeds are NOT a label problem.** 37 of the 47 are found neither by SHIVA nor by another seed, 38 of the 47 are found by no other seed. Our sensitivity gap is therefore real and stable -- it lies with the data (z-resolution, 5v) or with the method, not with the reference having over-marked. This rules out the convenient, obvious explanation.
**(3) On average, false positives look DIFFERENT from hits:** 28 voxels [18; 44] vs. 47 [29; 79], network probability 0.70 vs. 1.00; location lobar 36 / deep 11 / infratentorial 6 vs. 48 / 33 / 13. The majority of the false positives are thus smaller, less certain, and more often lobar -- no evidence against the reference. The 16 multiply confirmed ones are the interesting subset.
**What this means for the project's numbers:** the possible label share (up to +0.076 F1) is of the same order of magnitude as the seed noise (0.08, Section 3). Together, both mean: **differences below roughly 0.08 are fundamentally not interpretable in this project** -- neither against the noise nor against the uncertainty of the reference. This retroactively supports every decision in this project that came out as "not confirmed."
**Limitations, clearly stated:** SHIVA and our networks are NOT independent in the strict sense -- both learn from similar images and can respond to the same mimics (vessel cross-sections). Agreement is convergent evidence, not proof. The number 0.722 must not be reported as "our true performance," only as the upper bound of a proxy procedure. **The actual measurement remains open and needs a clinician** -- the package for it is ready and waiting with the user (5b).

### 5y Raising the ceiling does not help -- union candidates, measured (23 Sep 06:23; `mb-arena/stage2/pooled_union3_0922.json`)
**Prediction from 5w:** the union of the candidates from three seeds at threshold 0.05 raises the chain's ceiling from 0.851 to 0.927 (120 instead of 103 reachable bleeds) and quadruples the classifier's training set (703 instead of 278 VALDO candidates). The counter-argument was also stated there: a more sensitive network with a higher ceiling (blob, 0.884) had made the chain WORSE on 2026-09-21.
**Result, paired against the same second stage on the old candidates:** VALDO **0.584 vs. 0.623 (-0.039 [-0.106; +0.027])**, catalina-T2* 0.672 vs. 0.672 (+0.000), catalina-SWI 0.676 vs. 0.670 (+0.006). The simple chains on the same candidates also drop: threshold only 0.522 vs. 0.565, threshold + CSF rule 0.604 vs. 0.627. **The ceiling rose by 0.076, the achieved F1 fell by 0.039.**
**Read:** the counter-argument won. Of the 17 additionally reachable bleeds, the classifier does not pick out any, but it has to sort 409 additional false candidates instead (581 instead of 172) -- and the 378 candidates found by only ONE run are mostly exactly what a single run falsely sees. The union therefore mainly collects the seed's errors, not the seed's finds. **This sharpens the rule from 5s:** a higher ceiling only helps if the additional candidates are SEPARABLE; for candidates seen by only one run, that is precisely not the case. A cheap pre-test for this exists: the `n_runs` field from `merge_candidates` -- among the candidates with n_runs = 1, there are 378 items and only few true bleeds.
**Not measured, but obvious:** the union of ONLY the candidates found by at least two runs (325 instead of 703). This keeps part of the ceiling gain and discards the loners. Costs 40 min GPU; the ceiling of this subset can be computed without a GPU and should be checked BEFORE the run (rule from 5s).
**Addendum 28 Sep -- measured (23 Sep, `mb-arena/stage2/pooled_union_min2_0923.json`):** union of only the candidates found by at least two runs: CNN VALDO **0.608 vs. 0.623** (-0.015), catalina-T2* 0.672 vs. 0.672, catalina-SWI 0.677 vs. 0.670 (+0.007). Nothing above the noise of the second stage. **The ceiling path is closed:** neither all the loners nor the majority candidates give the classifier separable additional finds.
**Control:** the same second stage on the old candidates yields 0.623 / 0.672 / 0.670 vs. 0.625 / 0.672 / 0.670 from 21 Sep -- the GPU spread of the second stage here is 0.002, markedly less than the 0.03 from 5r.

### 5z Teacher-student as the second stage (user request 22 Sep; 23 Sep 06:23; `mb-arena/stage2/pooled_teacher_student_0922.json`)
**Original:** MicrobleedNet trains its classifier by knowledge distillation from a teacher network. In our case, the second stage has so far learned from hard 0/1 labels -- a gap in the deviations table that the user found.
**Our adaptation (`classifier.fit_predict_cnn(..., soft=, alpha=)`):** a teacher that sees more than the student would be cheating; a teacher trained on the same candidates it is teaching would just pass on rote-learned material. The nested evaluation, however, already delivers honest teacher probabilities OUTSIDE the respective fold: the inner models predict the held-out training candidates (used so far only for threshold selection and discarded afterward). The student learns from `(1 - alpha) * hard label + alpha * teacher`. Intended benefit: with 106 positive VALDO candidates, the hard target forces the network to memorize borderline cases as certainties.
**Check (CPU):** without soft targets, bit-identical to the old model; soft targets equal to the hard labels change nothing; alpha = 0 falls back exactly to the original; wrong length is rejected.
**Result, paired against the same CNN on the same candidates:**
| alpha | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| 0.3 | +0.010 [-0.008; +0.039] | **+0.010 [+0.001; +0.018]** | +0.003 [-0.006; +0.011] |
| 0.5 | +0.009 [-0.006; +0.033] | +0.006 [-0.006; +0.015] | +0.002 [-0.009; +0.014] |
| 0.7 | +0.005 [-0.011; +0.027] | +0.005 [-0.006; +0.013] | +0.002 [-0.014; +0.019] |
**Read, cautiously:** **all nine comparisons are positive** -- that is notable, but the values lie between +0.002 and +0.010 and are thus below the training noise of the second stage (0.03, 5r). Only T2* at alpha = 0.3 narrowly excludes zero, and that is one of nine comparisons without correction. The signs are also not independent (same folds, same candidates, nested alpha values). **Decision: not yet adopted.** The direction is right, the magnitude is not established. It could be established with the method the user himself proposed: 10 seeds per condition (~6 GPU hours for alpha = 0.3 against the plain CNN). This is the first condition in this project for which this effort is worthwhile -- because it is consistently positive and has a mechanism, not because a number looks big.
**Addendum 28 Sep -- the 10 seeds have been computed (23 Sep, `mb-arena/stage2/ts10_0..9.json`, ~49 min GPU each, ~8 h total) and were only evaluated today:** alpha = 0.3 against the same CNN, difference per seed -- VALDO **+0.001** (sd 0.009, 4/10 positive), catalina-T2* **+0.003** (sd 0.004, 8/10 positive), catalina-SWI **+0.002** (sd 0.005, 7/10 positive). Absolute: CNN 0.632 / 0.671 / 0.672, teacher-student 0.633 / 0.674 / 0.673. The direction from the first run holds on T2* (8 of 10), the magnitude is ≤ 0.003 -- less than the seed spread of the second stage itself (sd 0.009 / 0.003 / 0.004). **Decision: not adopted, closed.** A side finding that remains: the seed spread of the second stage is now measured with 10 runs (VALDO sd 0.009, catalina 0.003-0.004) -- much smaller than the 0.03 from 5r, which came from two runs with different candidate sets. The ten-seed measure applies for future comparisons of the second stage.
### 5aa What helps against the cohort difference -- first preliminary test without GPU (user question 23 Sep 06:34)
**Starting point (5n):** the transfer fails because of the IMAGES, not because of the labeling style -- catalina-T2* lies on a different intensity level (brain median 0.52 vs. 0.71 on the inverted scale), both catalina cohorts have
weaker lesion contrast, catalina-T2* has 5 mm slices. The obvious first suspicion: the scale. We divide by the MAXIMUM in the brain (the original's procedure) -- a single bright outlier thereby shifts the whole cohort.
The percentile scale p99.5 (`--norm p995`) was measured on 2026-09-20 WITHIN VALDO (+0.025, one seed, not confirmed) -- i.e. where the scales already match anyway. For the TRANSFER it had never been tested.
**Preliminary test (CPU, 3 min, 20-25 cases each; contrast here = difference bleed minus ring on the inverted scale, NOT the relative darkening from 5n -- the three contrast definitions must not be compared, section 11):**
| Cohort | Brain median today (scale max) | with p99.5 | Lesion contrast today | with p99.5 |
|---|---|---|---|---|
| VALDO | 0.706 | 0.726 | 0.117 | 0.124 |
| catalina-T2* | 0.522 | 0.555 | 0.095 | 0.102 |
| catalina-SWI | 0.692 | 0.739 | 0.062 | 0.066 |
**Result: the percentile scale does NOT close the gap.** It raises all three cohorts by the same small factor (3-7%); the gap between catalina-T2* and VALDO remains 0.52 vs. 0.73. The level is therefore not an outlier problem, but
lies in the distribution itself (tighter HD-BET mask, different tissue mix in the denominator). **This path is thus settled without a GPU run** -- 3 CPU minutes instead of 2 GPU hours, the same rule as with the robustness test (5v) and the
candidate upper bound (5s).
**What remains, in order:** (1) Aligning to the BRAIN LEVEL instead of to an extreme value -- z-normalization in the brain or median alignment. This targets exactly the measured quantity (0.52 vs. 0.71). The z-normalization was rejected on 2026-09-19 as
harmful (-0.060), but in a PACKAGE with augmentation and only WITHIN VALDO, where there is nothing to align. For the transfer it is unmeasured. Cost: one training run (~1.7 h) plus zero-shot (~12 min).
(2) Operating point per cohort, chosen by a MEASURABLE image property (slice thickness from the file header, brain median) instead of by performance -- thus leak-free. Unmeasured. (3) Histogram matching to a reference distribution before the
prediction, the classical route of domain adaptation; more expensive than (1) and with no advantage over it as long as (1) is unmeasured. **Not feasible:** the lesion contrast itself and the slice thickness are physics.
**All candidates measured (23 Sep 08:02; `cmb/analysis/harmonisation_check.py`, new; `ergebnisse/analysis/harmonisation_check_0923.json`; 12 cases per cohort, CPU 3 min).** What is reported is the SPAN between the three cohorts
after the respective normalization -- small means aligned. The decisive column is the lesion contrast: that is what the network detects.
| Normalization | Span brain level | Span brain spread | Span lesion level | **Span contrast** |
|---|---|---|---|---|
| max (today) | 0.188 | 0.038 | 0.199 | 0.060 |
| p99.5 | 0.194 | 0.039 | 0.182 | 0.057 |
| z in the brain | 0.338 | 0.186 | 0.331 | **0.362** |
| Median / IQR | 0.000 | 0.000 | 0.151 | 0.194 |
| white matter as anchor | 0.093 | 0.099 | **0.017** | 0.085 |
| **Landmarks (Nyul-Udupa)** | **0.000** | **0.000** | 0.080 | **0.018** |
**Two findings, one of them against my expectation.** (1) **Landmark alignment wins clearly:** level and spread exactly matched (by construction) AND the contrast span falls from 0.060 to 0.018, i.e. by a factor of three. It is
the only method here that can fix a NONLINEAR difference -- and apparently the difference is nonlinear. (2) **The z-normalization is the WORST of six** (contrast span 0.362, six times worse than today),
and Median/IQR is likewise worse than doing nothing at all (0.194). The reason is the same mechanism: the spread in the brain is determined by cohort-specific structure and noise; dividing by it aligns the DISTRIBUTION and
in doing so distorts the LESION CONTRAST differently for each cohort. This retrospectively explains the finding from 19 Sep (-0.060 in the package): the z-normalization there was probably not innocent, but the harmful part.
**Consequence for the queued runs:** [193]-[195] (z-normalization, training + two zero-shots) remain queued, but with a changed role -- **no longer as a promising candidate, but as a falsification test of this preliminary check.**
If the preliminary check says "z is the worst of six" and the GPU run confirms it, the tool is calibrated for future decisions (2 GPU hours for that are well spent). If the run says the opposite, the preliminary check is measuring
the wrong quantity and must be reconsidered. **The actual candidate now is landmark alignment** -- it still needs code in `turnier.py` (`--norm landmarks` with stored reference landmarks) and in the zero-shot.
**Two 0-1 variants remeasured (user question 23 Sep 11:01; `ergebnisse/analysis/harmonisation_check_0923b.json`):** `minmax` (darkest brain voxel to 0, brightest to 1) yields span 0.188 / 0.038 / 0.199 / **0.060** -- IDENTICAL DIGIT FOR DIGIT
to today's scale, because by construction it already is min-max (see section 11). The robust version `p1_p99` (1st and 99th percentile instead of the extremes, clipped to [0,1]): 0.171 / 0.038 / 0.161 / **0.063**, i.e. for contrast
minimally worse than today. **With that the ranking is complete:** landmarks 0.018 | p99.5 0.057 | max = min-max 0.060 | 0-1 robust 0.063 | white matter 0.085 | Median/IQR 0.194 | z 0.362. **All AFFINE scales land
between 0.057 and 0.063** -- that is the limit of what a linear mapping can achieve here, and it is broken only by the piecewise-linear landmark alignment (factor three). This is at the same time the strongest argument for
the claim that the cohort difference is NONLINEAR.
**Limitation:** the preliminary check measures image statistics, not network performance. It can only exclude and rank, not prove -- precisely why the falsification test.
**User question 23 Sep 11:58: "can we repeat all experiments with normalized inputs?" -- the preliminary check says: only the cross-cohort ones, and the rest would CAUSE HARM.** The same measurement, this time with the THREE
VALDO sub-cohorts as "cohorts" (`ergebnisse/analysis/harmonisation_innerhalb_valdo_0923.json`, reference valdo-2):
| within VALDO | span level | span spread | span lesion contrast |
|---|---|---|---|
| today (min-max) | 0.048 | 0.165 | **0.039** |
| Landmarks | 0.000 | 0.000 | **0.127** |
Per sub-cohort the lesion contrast falls from 0.113 / 0.104 / 0.074 to 0.149 / 0.098 / **0.023** -- the thick-slice cohort 3 loses two thirds of its contrast.
**The explanation, and a rule follows from it:** the three VALDO sub-cohorts already agree in LEVEL (0.675 / 0.722 / 0.705), but differ strongly in SPREAD (0.086 / 0.143 / 0.250). Landmark alignment forces
equal percentiles and thereby compresses the widely spread cohort 3 -- its bleeds lose the contrast they had against their own surroundings. **Landmarks help where cohorts differ in LEVEL
(VALDO vs. catalina: 0.71 vs. 0.52), and harm where they differ in SPREAD (within VALDO).**
**Prediction, recorded BEFORE the result of the running run:** the landmark run will be WORSE than 0.646 within VALDO, and most strongly so in cohort 3 (3-4 mm, today 0.479). If that holds, the preliminary check is confirmed for the
second time and may exclude runs in the future. For the zero-shot on catalina, on the other hand, I still expect a gain -- there it acts on the level.
**So the answer to the user question:** repeating all 35 steps of the decision ladder would be over 50 GPU hours, and the preliminary check says that the inputs within VALDO would thereby be WORSE aligned than today --
the twelve negative training findings would therefore be repeated not more fairly, but more unfairly. Only the CROSS-COHORT experiments (zero-shot, mixing, fine-tuning, mix-all) make sense, and even those only once the running run
shows a gain. The order is thus fixed: first the result, then the decision -- not the other way around.
**User note 23 Sep 11:10: "min-max could help especially when mixing data."** Min-max itself is a no-op here -- remeasured on ALL THREE grids (minimum in the brain 0 to 1.8e-5, maximum 0.9955; a
min-max stretch would be a factor of 1.004). The idea behind it, however, hits an unmeasured point: **all our mixing experiments ran on the unharmonized scale.** Mixed at equal budget per case is today on par with dedicated
training (5h, step 25) -- and exactly during mixing the network sees both distributions WHILE training, that is where alignment should help most. **Queued:** mixed VALDO + catalina-T2*, 120 epochs, with
landmark alignment, against the existing mixed run on the old scale (catalina-T2* 0.641, VALDO share 0.582). **Set in advance:** a gain only counts above the seed noise of 0.08 (section 3); an improvement on BOTH
shares is expected, since the alignment relieves the network of the task of serving two scales at once.
**For this `turnier.py` had to be changed (backup .vor-landmarken-0923):** `--manifest` and `--norm` used to be mutually exclusive, because the manifest loader silently overwrote the normalization loader -- the lock was a symptom, not
the cause. Now the normalization is applied LAST and wraps the loader actually used. CPU test with both cohorts at once: all four cases hit the reference percentiles to six decimal places.
**Built (23 Sep 08:58):** `turnier.py --norm landmarken` (backup .vor-landmarken-0923), `cmb.transfer.zero_shot --norm landmarken`, and `code/landmarken_bauen.py`, which generates the reference as a file -- one file, so that training and
prediction demonstrably use the same scale, even for a cohort added later. Reference from the 57 CV cases: P10 0.6278, P25 0.6619, P50 0.7064, P75 0.7939, P90 0.9514, P99 0.9766 (`dev/landmarken_valdo.json`).
**Disclosed:** the intensity distribution of every CV case flows into this reference at 1/57, so including that of the respective test case. These are pure percentiles without labels -- the usual approach in the literature and a very small
influence, but an influence; `landmarken_bauen.py --faelle` allows a leak-free version per fold, should a reviewer insist on it. **Check (CPU, three cases):** after alignment all cases hit the reference percentiles to
five decimal places exactly (before they were at P10 0.33-0.39 instead of 0.63), the background stays exactly 0, no brain voxel becomes 0, the label is untouched. **Queued:** training with landmarks plus zero-shot on both catalina cohorts;
reference are the existing zero-shot values (T2* 0.622, SWI 0.305) and as the noise level the zero-shot of the base with seed 1 (0.638 / 0.311, WORKLOG.md 5p). **Set in advance:** the alignment counts as a gain if it is, on at least
one catalina cohort, paired above zero AND greater than the gap between the two base seeds (0.016 / 0.006); within VALDO a tie is expected, since there is nothing to align there.

### 5ab Pretrained encoder for the second stage (user request 22 Sep; result 23 Sep 09:45; `mb-arena/stage2/pooled_vortraining_0923.json`)
**Idea (literature list item 5, transferred to our own data):** the second stage has too few labels (106 positive VALDO candidates), but enough images -- every cohort supplies patches, and for pretraining none of them needs
a label. Instead of downloading foreign weights (architecture doesn't fit, provenance unclear), the encoder FIRST learns to reconstruct cut-out cuboids within its own patch, and is then fine-tuned on classification.
**Why gap filling specifically:** it is the one task that teaches exactly what separates our two classes -- a vessel CONTINUES, a bleed stops. Anyone who wants to fill a gap must learn how local structure
continues. Intensity tasks were deliberately left out: the robustness test (5v) shows that the network is already invariant to gamma, contrast and noise -- there is nothing to learn there.
**Leak-free:** the pretraining sees only `train_x`, i.e. the training folds; nothing from the test fold enters, not even without a label.
**Preliminary check (CPU, real candidate patches):** the reconstruction error on UNSEEN patches falls from 0.332 (random weights) to 0.0049 after 12 epochs; the trivial baseline (guessing the mean everywhere) is at
0.105. The encoder is thus learning real structure, not just the brightness level -- the pretext exercise works.
**Result, paired against the same CNN on the same candidates:**
| Variant | VALDO | catalina-T2* | catalina-SWI |
|---|---|---|---|
| 60 epochs pretraining | +0.012 [-0.009; +0.040] | -0.005 [-0.015; +0.005] | -0.011 [-0.027; +0.003] |
| 20 epochs pretraining | -0.000 [-0.052; +0.050] | -0.005 [-0.016; +0.005] | -0.009 [-0.029; +0.011] |
**Read as: no gain, and the sign flips per cohort.** On VALDO (the most data-poor cohort with 278 candidates) +0.012, on both catalina cohorts slightly negative -- no interval leaves zero, all values lie far
below the training noise of the second stage (0.03, 5r). **Interpreted:** the pretext exercise does learn something measurable (reconstruction error 68 times better than random), but apparently nothing that improves the separation of bleed/mimic
-- being able to continue structure does not mean being able to distinguish vessel from bleed. On top of that comes a price paid: the three members now start from the SAME pretrained encoder and differ only
in head initialization and data order -- the ensemble becomes less diverse. This is likely to eat back up the small VALDO gain on the two other cohorts. **Decision: not adopted.**
**Limitations:** one pretext task, one architecture, one seed per variant; what was tested was reconstruction, not contrastive learning (SimCLR-style) or rotation prediction. The statement is: THIS pretext exercise brings nothing here -- not
that pretraining brings nothing in principle.

### 5ac Intensity alignment measured -- landmarks harm everywhere, z-norm is neutral, the preliminary check from 5aa did NOT hold up (runs 23/24 Sep, evaluated 28 Sep; `ergebnisse/analysis/operating_point_valdo_norm_0928.json`, `valdo_norm_split_0928.json`, `mb-arena/transfer/auswertung_zeroshot_{t2star,swi}_norm_0928.json`, `auswertung_t2star_mix_e120_nlm_0928.json`)
**Why only now:** the four runs (training with `--norm landmarken` / `--norm z`, two zero-shots each, mixed e120 with landmarks) were finished on 24 Sep 03:49, the evaluation was missing -- since 25 Sep `claude -p` has been failing in the morning check (rc=1, login), nobody had read the results. Made up on 28 Sep on the CPU (four evaluations, ~20 min).
**Result, read against the rules set in advance in 5aa (zero-shot gain only counts if > 0.016 / 0.006 = gap between the base seeds s42/s1; a tie expected within VALDO):**
| Scale | VALDO fixed cell (CSF rule) | VALDO cross-wise, paired against base | catalina-T2* zero-shot, paired against s42 | catalina-SWI zero-shot, paired against s42 |
|---|---|---|---|---|
| today (max), s42 | 0.646 | 0.622 | 0.622 (P 0.60 S 0.65 FP 4.76) | 0.305 (P 0.32 S 0.29 FP 6.72) |
| Noise level: base s1 | 0.590 | 0.564 | 0.638, +0.016 [+0.003; +0.033] | 0.311, +0.006 [-0.015; +0.030] |
| **Landmarks (Nyul-Udupa)** | **0.549** | 0.559, **-0.064 [-0.143; +0.013]** | **0.574, -0.048 [-0.068; -0.027]** | **0.271, -0.034 [-0.058; -0.012]** |
| z in the brain | 0.615 | 0.618, -0.005 [-0.068; +0.066] | 0.618, -0.004 [-0.035; +0.031] | 0.283, -0.022 [-0.050; +0.006] |
Per VALDO sub-cohort (raw, `--split-valdo`): landmarks **-0.194 [-0.337; -0.010]** (cohort 1, 4 mm: 0.557 -> 0.364), **-0.078 [-0.155; -0.016]** (cohort 2, 0.8 mm), **-0.138 [-0.253; -0.016]** (cohort 3, 3-4 mm: 0.479 -> 0.341); false positives per case 1.5 -> 3.9, 1.7 -> 2.7, 1.2 -> 2.1. z-norm: -0.123 n.s. / -0.002 / +0.021, all intervals around zero.
Mixed VALDO + catalina-T2*, 120 ep., WITH landmarks against the old mixed run: catalina-T2* **0.630 vs. 0.641** (against dedicated training 0.663: -0.032 [-0.057; -0.009], confirmed), VALDO share **0.355 vs. 0.582** -- the alignment did not relieve the network of the work during mixing, it ruined the VALDO share.
**Read in three sentences.** (1) Not a single one of the four comparisons set in advance is a gain; landmarks are CONFIRMED worse on both catalina cohorts and in all three VALDO sub-cohorts, with two to three times as many false positives. (2) The prediction from 5aa was only half right: worse within VALDO -- yes (but most strongly in cohort 1, not 3); zero-shot gain on catalina -- **no, the opposite.** (3) The z-normalization, which the preliminary check ranked as the WORST of six scales (contrast span 0.362 vs. 0.060), is in reality neutral everywhere; the landmarks, which it ranked as the BEST (0.018), are confirmed worse everywhere. **The ranking of the preliminary check is thereby falsified.**
**What this means for the tools (more important than the single result):** `cmb/analysis/harmonisation_check.py` measures the span of the MEAN lesion contrast BETWEEN cohorts. That is apparently not the quantity the network depends on: the network sees, per case and per lesion, the contrast against its immediate surroundings, and a piecewise-linear alignment with knots at P10..P99 stretches or compresses the tail below P10 -- that is where the bleeds sit -- DIFFERENTLY PER CASE. Equal cohort means with per-case shifted contrast is exactly what costs a network more false positives. The tool describes image statistics correctly (that was checked) and nevertheless draws the wrong conclusion about network performance. **Rule withdrawn:** the preliminary check may NOT exclude any runs and may not queue any; it remains as a description. The other two CPU preliminary checks (robustness test 5v, candidate upper bound 5s/5w) have each been confirmed once and each failed once (5y) at their limit -- they too are ordering aids, not deciders. The lesson is recorded under [[fehler-heisst-geruest]]: the falsification test achieved exactly what it was queued for, just with the opposite sign.
**With that the intensity-scale path is CLOSED:** all affine scales are equivalent (5aa, p99.5 within VALDO +0.025 n.s.), the two non-affine or distribution-matching ones lose or stay equal. The VALDO/catalina cohort difference does not lie in the scale. What remains is physics (5 mm slices, weaker contrast) and the labeling style -- and for both, dedicated training per cohort (0.663 / 0.670 with second stage) is the measured best answer.
**Not adopted:** `--norm landmarken`, `--norm z`. Both switches remain in the code, `dev/landmarken_valdo.json` remains in place as evidence.

### 5ad Freezing and the ONE-TIME measurement on the 15 held-out cases (user decision 28 Sep ~15:00; set in advance BEFORE the run)
**Frozen chain for VALDO (nothing in it will be changed after this measurement):** first stage a03-aniso, recipe r3 on `gitter_d`, 60 epochs, intermediate-checkpoint selection on the inner fold, seeds 42 and 1 (`ergebnisse/turnier/a03-aniso-e60-gd`, `a03-aniso-s1-e60-gd`); per seed the mean of the five fold models (this is how it is delivered, `cmb.transfer.zero_shot`), then the mean of the two seeds (`cmb.analysis.ensemble`); post-processing fixed cell threshold 0.3 / at least 2 mm3 / giant components removed; CSF rule (majority label SynthSeg = free CSF or ventricle -> discard). The second stage (shared 3D-CNN) belongs to the chain for catalina-SWI; on VALDO it was on par with the CSF rule (5e, 0.623 vs. 0.627) and is NOT applied here -- the measurement concerns the VALDO delivery as it was measured in the CV (two-seed ensemble with CSF rule 0.647, section 5u).
**The 15 cases:** `dev/manifest_valdo_gitter_d_heldout.json` (new switch `build_valdo_manifest --held-out`; the CV manifest was rebuilt in the process and is identical except for the order; the 15 cases carry the AUXILIARY fold 0 in the manifest, because all loaders only accept folds 0-4 -- the first start attempt failed on this with a KeyError; the fold is just a folder name, nothing is trained). 97 reference bleeds, of which **73 in one case (sub-110, cohort 1)**, 11 in sub-221, three cases without a bleed, nine cases with 1-3. These cases were already touched ONCE on 14 Sep (we5, old model r3 with majority vote: F1 0.19; section 6, 10e) -- this here is the second and last touch. No model, no threshold, no rule has been chosen on these cases since 14 Sep.
**Expectation, recorded before the result:** pooled F1 with CSF rule between 0.55 and 0.70. Reasoning: the CV number is 0.647 (ensemble) resp. 0.605 ± 0.034 (individual seeds), and the class ">5 bleeds" was at 0.72 in the CV -- sub-110 carries three quarters of the reference, so the pooled F1 is essentially the F1 of this one case. Therefore three further numbers are set IN ADVANCE and reported: (a) pooled without sub-110 (24 bleeds, 14 cases), (b) median of the F1 per case (challenge rule, `subject_level_f1`), (c) false positives per case in the three bleed-free cases. A deviation downward below 0.50 I would read as overfitting of the chain to the CV (many comparisons on the same 57 cases, section 6); one upward above 0.70 as a lottery win through sub-110, not as evidence.
**What does NOT happen:** no second attempt, no cell is chosen after looking at this number, no seed added afterward. The result goes into the paper as is, with interval.
**Execution:** `cmb.transfer.zero_shot` per seed (GPU via `grossauftrag --vram 20`), `cmb.analysis.ensemble`, then `cmb.transfer.evaluate` (raw), `cmb.analysis.operating_point` (only the fixed cell with CSF rule counts; the cross-wise choice is meaningless with a single fold) and `cmb.analysis.subject_level_f1`. Result in `ergebnisse/validierung/heldout_0928/`.
**RESULT (28 Sep 16:10, GPU 377 s for both seeds, evaluation CPU; `ergebnisse/validierung/heldout_0928/summary.json`, model checksums `FROZEN.sha256`):**
| Measurement | Ensemble s42+s1 | s42 | s1 |
|---|---|---|---|
| pooled, 15 cases, 97 CMB, fixed cell + CSF rule | **0.468** | 0.472 | 0.468 |
| pooled raw (P / S / FP per case) | 0.463 (0.68 / 0.35 / 1.07) | 0.480 | 0.456 |
| (a) pooled WITHOUT sub-110, 14 cases, 24 CMB, with CSF rule | **0.731** (raw 0.679, P 0.59 S 0.79) | 0.704 | 0.731 |
| (b) challenge rule, median F1 per case (12 cases with bleed), with CSF rule | **0.667** (mean 0.591; mean over all 15: 0.540) | 0.667 | 0.667 |
| (c) bleed-free cases (3) | 1.0 false positive per case raw, 0.73 with rule; 1 of 3 completely without false positive | | |
| sub-110 only, 73 CMB | raw F1 0.330, P 0.83, **S 0.20**, 3 false positives; with rule 0.315 | | |
**Read against the pre-registration.** The pooled F1 at 0.468 is BELOW the range of 0.55-0.70 named in advance -- and the number (a) fixed in advance fully explains why: without sub-110 the chain is at 0.731, i.e. at the UPPER end of the CV numbers (0.647 ensemble, 0.605 ± 0.034 individual seeds). The median per case by the challenge rule (0.667) is above the CV value of the same rule (0.56 with CSF rule, section 5c/5a) and close to the challenge top score (0.67, three sequences). The chain is thus NOT overfitted to the CV -- on 14 of the 15 cases it holds what the CV promised, and more.
**What sub-110 is:** the most severe case in the entire dataset (73 bleeds; the three most severe CV cases in cohort 1 have 12 / 6 / 6, in the whole CV set no case has more than ~30). Its bleeds are small (median 10 mm3, 35 of 73 under 10 mm3) and low-contrast: of the 36 lesions below the contrast median the network finds 1 (sensitivity 0.03), of the 37 above it 14 (0.38); 34 of the 46 missed lobar lesions carry a maximum probability below 0.05 -- they are not narrowly missed, but not seen at all (`missed_lesions_nur110.json`). 12 lesions lie, according to SynthSeg, within the CSF label (sensitivity there 0.08). Precision on this case is 0.83: what the network reports is correct. This is exactly the picture from 5c/5i for small, low-contrast bleeds -- here 73-fold in a single case.
**How this goes into the paper (set in advance: like this, with interval, no second attempt):** both numbers side by side -- pooled 0.468 over 15 cases, of which one case carries 75% of the reference and sensitivity 0.20; without it 0.731; median per case 0.667. The honest statement: the chain holds up on ordinary cases (0-11 bleeds), and it fails under a massive burden of small, low-contrast bleeds of a kind that did not occur in training (cohort 1 has only three cases over 5 in the CV set). The sensitivity for such cases is the open limit of the method, not a measurement error. **With this the VALDO chain is frozen and measured.** Addendum section 6 and 10e.
### 5ae Research line after the freeze: three arms, fixed in advance (user task 28 Sep ~17:00 "start a research"; research `research/options-2026-09-28.md`)
**Starting point:** the VALDO chain is frozen and measured (5ad); the 15 cases are used up. Everything further is measured ONLY in the 5-fold CV on the 57 cases, paired, with two seeds (noise level 0.08 for single seeds, Section 3). Standard: different information or a different training distribution, not more sensitivity (5q, 5ac).
**Arm 1 -- artificial bleeds and confounders in the FIRST stage** (CenSynCMB 2026: on VALDO 71.7 -> 74.3 lesion F1 as the last step of their chain; our gap: the network has almost no examples for a massive burden of small, low-contrast bleeds, 5ad).
Implementation `cmb/synthesis/build_synthetic_grid.py` -> `dev/gitter_s`: per CV case, N ~ U{2..6} artificial bleeds and M ~ U{3..8} vessel confounders are placed into the image from `gitter_d`, FRST is recomputed NEW as in the grid construction, the label extended around the artificial bleeds (confounders stay 0). All parameters come from the MEASURED distribution of the 139 real bleeds (`ergebnisse/analysis/lesion_stats_valdo_0928.json`, `cmb/synthesis/lesion_stats.py`): volume per cohort via bootstrap from the real values up to the 75th percentile (median 10 mm3; cohort1 12.5, cohort2 7.3, cohort3 16), peak contrast per cohort from the real distribution (median 0.20; cohort3 0.12), tissue location according to the real proportions (cortex 29%, white matter 30%, deep nuclei 27%, infratentorial 11%; no CSF), at least 6 mm distance to real and other artificial lesions, 3 mm to the brain edge. Shape: Gaussian ellipsoid (half-value volume = drawn volume, in-slice anisotropy 0.8-1.25). Slice thickness: the ellipsoid is averaged in z per source-slice block (4 mm cohort1, 0.8 mm cohort2, 3-4 mm cohort3, from the case's `info.json`) and linearly back-interpolated -- the same mechanism as acquisition + grid construction; the label fills the whole block as in the nearest-neighbour grid construction. After averaging, the peak contrast is scaled to the drawn value so that the contrast distribution AFTER processing matches the real one. Confounders: Gaussian tubes (random direction, length 6-16 mm, half-width 0.8-1.6 mm, contrast 0.7-1.0 of the drawn value), location cortex/white matter. Additive on the inverted scale, capped to [0, 1], brain mask unchanged.
Training: new switch `turnier.py --synthese dev/gitter_s` (folder suffix `-syn`): a sampling wrapper like `mit_harten_negativen` swaps image/FRST/label for the ARTIFICIAL version ONLY for the training cases of the fold, draws the sample, and restores the real data. Test and inner selection fold never see an artificial voxel; without the switch the code is untouched. Side effect, disclosed: the lesion sample (60%) draws uniformly over real AND artificial components -- with ~4 artificial against 2.4 real per case, about 60% of the lesion patches are artificial.
**Arm 2 -- cubic interpolation of the image** (4.8, open since 19 Sep): `dev/gitter_k` = `gitter_c_bauen.py` with `GITTER_C_ORDNUNG=3` from `preproc_roh` (the linear version was previously rebuilt bit-identical to `gitter_d` on sub-201), label and brain mask identical to `gitter_d`, image capped to [1e-6, 1]. The conversion costs 15-30% lesion contrast linearly; cubic should recover part of that. Per 5ac, NO CPU contrast measurement is run beforehand to decide the run -- it is only carried along for description.
**Arm 3 -- Momeni/CSIRO SWI dataset** (75 SWI, 175 real + 750 artificial bleeds, coordinates only, CSIRO Data Licence, free: doi 10.25919/AEGY-NY12, data.csiro.au/collection/csiro:50304): read the licence, download, check image parameters; goal = first INDEPENDENT public test of the SWI model (center-distance metric, because there are no masks) and possibly training growth for SWI. No GPU effort before inspection.
**Runs (queued, ~1.1 h GPU each):** Arm 1: a03-aniso e60 gitter_d + synthesis, seeds 42 and 1. Arm 2: a03-aniso e60 gitter_k, seeds 42 and 1. Reference: base gitter_d s42 0.585 raw / 0.646 with CSF rule, s1 0.553 / 0.590 (Section 3, 5u); two-seed ensemble 0.647.
**Decision rule, fixed in advance:** an arm counts as a win if (i) BOTH seeds are paired above their respective base and (ii) the arm's two-seed ensemble paired against the base's ensemble has an interval above zero (fixed cell 0.3 / 2 mm3 with CSF rule, `operating_point` + `ensemble`). Secondary measures named in advance: sensitivity for bleeds under 10 mm3 and for the class ">5" (that's where the weakness from 5ad lies), false positives per case. An arm that buys sensitivity and loses F1 (pattern of blob/center, 5j/5k) counts as negative, even if the secondary measures rise. Expectation, recorded: Arm 1 raises the sensitivity of small bleeds; whether F1 follows depends on whether the confounder examples keep the false positives in check -- I give that 50%. Arm 2: small effect, probably positive in cohort 3, not interpretable below 0.03.
**Status 28 Sep 19:30 -- built, checked, queued:**
- `cmb/synthesis/lesion_stats.py` -> `ergebnisse/analysis/lesion_stats_valdo_0928.json` (139 real bleeds: volume p10/50/90 = 4.7 / 10 / 45 mm3; peak contrast 0.094 / 0.201 / 0.275; mean contrast 0.051 / 0.102 / 0.156; z-extent 2 / 3 / 6 slices; tissue cortex 40, white matter 41, deep nuclei 38, infratentorial 15, CSF 5).
- `cmb/synthesis/build_synthetic_grid.py` -> `dev/gitter_s`: 57 cases, all checks passed (real label preserved, brain mask the same, component count = real + artificial), 209 artificial bleeds, 319 confounders, seed 0, ~10 s per case. First draft had oversized labels four times for thick slices (block-filling added on top of the drawn volume); fixed: for thick slices the base area = volume / slice thickness, as with the real label. Re-measured with the same ring method on five test cases (25 artificial): cohort1 volume 9 / 9 / 15 (real 7 / 12.5 / 83), peak 0.17 / 0.19 / 0.27 (real 0.09 / 0.17 / 0.27), z 4 / 4 / 4 (real 4 / 4 / 8); cohort2 5.8 (real 7.3), peak 0.185 (0.228); cohort3 15.8 (16), peak 0.176 (0.124). Small and within the real contrast range, as intended; cohort3 somewhat more contrast-rich than real. Check figure `ergebnisse/analysis/synthese_qc/gitter_s_0928.png` (real and artificial bleeds side by side, confounders as short dark tubes).
- `turnier.py --synthese` (backup `.vor-synthese-0928`): wrapper test on 8 cases -- real data bit-identical after drawing, test cases never loaded, components per training case real -> artificial e.g. 0 -> 3, 2 -> 4, 3 -> 9, sample different, cache rebuilt on fold change.
- `dev/gitter_k` (cubic): 57 cases, label and brain mask bit-identical to `gitter_d`, mean image deviation in the brain 0.011, mean lesion intensity 0.797 -> 0.814 (description, not a decider).
- Queued: [203] gitter_k s42, [204] gitter_k s1, [205] gitter_d + synthesis s42, [206] ditto s1 -- ~1.1 h each; [203] running since 14:47. Evaluation as fixed in 5ae.
- Arm 3: CSIRO collection csiro:50304 has 4073 files / 17.7 GB (NoCMBSubject 313, rCMB_DefiniteSubject 57, sCMB_* 3700 synthetic variants, 3 xlsx with locations); licence "CSIRO Data Licence": non-commercial only, with attribution, NO redistribution -- only rCMB + xlsx are fetched to `~/data/extern/momeni-csiro-cmb/` (LIESMICH.md there). The S3 host `s3.data.csiro.au` cannot be resolved via the local resolver (Cloudflare CNAME), workaround `curl --resolve` with the address from 1.1.1.1.
- **Arm 3, status 28 Sep 21:00:** 57 files + 3 tables fetched (`rCMB_locations.json`: 146 bleeds in 57 SWI, median 1 per case, max 17; coordinates (x, y, z) with base 1 -- checked: at these voxels the image is 5.5 SD darker than the surroundings, every other axis order/base gives 0). Images 176 x 256 x 80, 0.94 x 0.94 x 1.75 mm, LPS, skull-free, bias-corrected and histogram-matched by the authors ("BFC_50mm_HM"); file names carry time points T0-T3 of the same subject (e.g. 122_T1 / 122_T2) -- the cases are NOT independent, intervals would need to be bundled per subject. Grid built with `cmb/transfer/build_momeni_grid.py` (same functions as VALDO/catalina; label = sphere r 1.5 mm around each location, substitute for missing masks): 57 cases, 146/146 lesions on the grid. Queued [207]-[209]: zero-shot of the catalina-SWI model (s42, s1) and, for comparison, the VALDO-T2* model. **Expectation, in advance:** SWI model raw 0.30-0.55 (own CV number 0.626 threshold only; foreign scanner, histogram matching, point labels with touch rule); VALDO model clearly below (on catalina-SWI 0.305). Evaluated primarily by the challenge rule (center distance 5 mm, `subject_level_f1`), because the reference is points; without CSF rule (no SynthSeg). Second stage not applicable (no application mode; only nested CV).
**Arm 4 -- pretrain classifier on VALDO, fine-tune on catalina (user idea 28 Sep ~18:00; fixed in advance BEFORE the run).**
Question: own U-Net per cohort (measured correct, 5h/5n), but the classifier NOT trained jointly on the pool, but pretrained on the 278 VALDO candidates and fine-tuned on the catalina training folds. Why this matters despite the same data: today's chain needs the private candidates during training and is not publishable; this chain would be (U-Net + classifier on VALDO public, clinic trains its own U-Net and only tunes the classifier). Clarification on the user's question: the joint 3D-CNN sees ONLY the patches (image, FRST, probability; `candidates.py` l.136), no tissue feature -- the tissue class only feeds into the feature models. His SWI gain comes from the fact that it doesn't need the hard CSF rule and learns the appearance from the larger pool (my statement in conversation "learns with the tissue feature" was wrong).
Implementation: `classifier.fit_predict_cnn_transfer` (same layers, patch, jitter, mirroring, three averaged members like `fit_predict_cnn`; pretraining 40 epochs lr 1e-3 on the source, fine-tuning 20 epochs lr 3e-4; pretraining once per seed, cached -- exact, because the source does not depend on the target fold) and `pooled.py --transfer valdo=t2star,swi --fractions 0 0.1 0.25 0.5 1`: per target fold k, fine-tuning is done on a CASE-WISE fraction of the catalina training folds (nested subsets, same permutation for all fractions), threshold from inner folds of these candidates; fraction 0 = pure VALDO classifier with threshold from VALDO's own inner folds (zero-shot of the second stage). Backups `.vor-transfer-0928`. Smoke test with 1 epoch on the CPU before queueing.
Runs: 3 seed offsets (0, 1, 2), each one call with `--models cnn` (joint CNN as reference in the same run, paired per case) and the five fractions; candidate sets S1-S3 (`-weich`). Estimated 1.5-2 h GPU per offset, behind [209].
**Expectation, in advance:** (1) fine-tuning with fraction 1 lies within ±0.01 of the joint CNN (same data, different order; noise level of the second stage sd 0.003-0.009 from 5z); (2) zero-shot (fraction 0) lies below that, clearly on SWI (VALDO never sees SWI patches; U-Net zero-shot on SWI was 0.305 against 0.626), little on T2*; (3) the learning curve reaches the level on T2* at 25% (T2* resembles VALDO), on SWI only at 50%. **Decision:** a fraction counts as "sufficient" if its mean over the three offsets lies within 0.01 of the joint CNN. The curve is evaluated, not a single value; a gain OVER the joint CNN is not expected and would only be credible with 10 offsets.
**INTERIM STATUS 28 Sep 21:00 (user request; rules unchanged; `ergebnisse/analysis/5ae_0929/zwischen_*.json`):**
| | Base s42 | cubic s42 | Base s1 | cubic s1 | Base Ens. | cubic Ens. | Synthesis s42 |
|---|---|---|---|---|---|---|---|
| fixed cell 0.3 / 2 mm3 + CSF rule | 0.646 | 0.573 | 0.590 | 0.606 | 0.647 | 0.611 | **0.467** |
| cross-selected, paired against base | 0.622 | 0.605, -0.018 [-0.079; +0.043] | 0.564 | 0.607, +0.043 [-0.014; +0.097] | 0.647 | 0.623, -0.024 [-0.086; +0.029] | 0.468, **-0.154 [-0.287; -0.038]** |
**Arm 2 (cubic) is thereby DECIDED: no gain.** Rule (i) violated (s42 below base), rule (ii) violated (ensemble interval around zero, point value negative). Notable only: the cubic network wants higher thresholds (0.5-0.7 instead of 0.3) -- the calibration shifts, F1 doesn't. The recovered contrast (lesion intensity +0.017) is apparently not a bottleneck. Cubic is not adopted.
**Arm 1 (synthesis), seed 42: clearly WORSE, and differently than expected.** Expected was "more sensitivity, maybe more false positives"; measured is LESS sensitivity (0.45 against 0.64) AND less precision (0.49 against 0.60). Per cohort raw: cohort1 -0.184, cohort2 -0.087, cohort3 -0.203; sensitivity of small bleeds (< 7 mm3) 0.52 -> 0.34, low-contrast (below median) 0.51 -> 0.30 -- exactly the group synthesis was supposed to help loses the most. Inner F1 (real selection cases) lower in four of five folds; training loss higher (0.142 against 0.102 in epoch 60), so not a trivial task, but one that pulls the network away from the real. Checked: the synthesis grid was loaded (33-36 training cases per fold per the run log), test and selection fold real. Rule (i) is thereby already violated, seed 1 (running, [206]) can no longer save the arm; it only delivers the noise level.
**Hypotheses, noted before diagnosis:** (H1) budget: 60% of the lesion patches are now artificial, the number of real lesion patches per epoch drops to ~40% -- per 5h, the budget per real lesion is decisive; (H2) the artificial bleeds are distinguishable from the real ones (smooth Gaussian shape on a noisy background, block-filled label on a soft image) and the network learns "artificial" as its own, easy class, while the real small bleeds become relatively rarer and harder; (H3) the confounders (bright tubes, label 0) raise the bar for small bright points in general. Diagnosis runs on the CPU: fold-0 models (base, synthesis) on three training cases of the synthesis grid -- maximum probability per real bleed, per artificial bleed, per confounder.
**Diagnosis 28 Sep 21:35 (`ergebnisse/analysis/5ae_0929/zwischen_syn_probe.txt`; three training cases of fold 0, one cohort each; fold-0 models base and synthesis; maximum probability per component):**
| Case | Model | real CMB | artificial CMB | Confounders |
|---|---|---|---|---|
| sub-103 (cohort1, 6 real / 3 artificial / 8 conf.) | Base | 0.59 0.62 1 1 1 1 | 0.67 1 1 | all 0.00 |
| | Synthesis | 0.22 0.62 1 1 1 1 | 0.85 1 1 | all 0.00 |
| sub-218 (cohort2, 26 / 5 / 3) | Base | 0 0 .25 .33 .41 .50 .50 .62 .75 .83 .90 .96 .99 1 ... | 0 0 0 .13 1 | all 0.00 |
| | Synthesis | 0 0 0 .25 .25 .33 .34 .39 .43 .50 .50 .50 .75 .86 .88 1 ... | .25 .51 .84 .87 1 | all 0.00 |
| sub-305 (cohort3, 2 / 3 / 8) | Base | .75 1 | .50 1 1 | 0 ... 0.50 |
| | Synthesis | 1 1 | .50 1 1 | 0 ... 0.23 |
**Read:** (H3 REJECTED) the confounders get 0.00 from BOTH models -- even from the base, which never saw one; bright Gaussian tubes are not hard negatives, the network never confused them with bleeds. (H2 only for cohort 2) the base finds almost all artificial bleeds in cohort1/cohort3 (0.5-1.0) without ever having seen one -- they are realistic enough there; in cohort2 (0.8 mm) 3 of 5 are invisible to the base (too small, too smooth, without the sharp edge of real bleeds at high resolution). (H1 SUPPORTED) on the SAME training cases the synthesis model gives real bleeds LOWER probabilities than the base (sub-218: 0.00/0.00/0.00/0.25/0.25/0.33 against 0.00/0.00/0.25/0.33/0.41/0.50) -- it has learned the real bleeds worse, because 60% of its lesion patches were artificial and the budget per REAL lesion (5h) has dropped.
**Arm 1, first version: NEGATIVE, closed.** Rule (i) violated; seed 1 ([206]) is running out and only delivers the noise level.
**Second version (5ae-1b), fixed in advance 28 Sep 21:40:** (1) budget of the real lesions UNCHANGED: the sample is drawn as in the base (60% real lesion / 25% brain / 15% arbitrary); afterward half of the patches WITHOUT a lesion are replaced by patches around artificial bleeds (`--synthese-anteil 0.5` -> 20% of all patches artificial, real lesion patches exactly as before); (2) no confounders; (3) flatter profile with sharper edge (super-Gaussian, `--profile flat`) against the invisibility in cohort2; (4) otherwise as in 5ae (N ~ U{2..6}, sizes/contrast/location from the real distribution). Grid `dev/gitter_s2`. Expectation: sensitivity of small bleeds rises slightly, F1 within the noise (±0.03); gain still counts only per rule (i)+(ii). Queued behind Arm 4.
**RESULT ARMS 1-3 (automatic evaluation 28 Sep 22:17, `ergebnisse/analysis/5ae_0929/`; read 22:45 against the rules above):**
| fixed cell 0.3 / 2 mm3 + CSF rule; paired = cross-selected | s42 | s1 | two-seed ensemble |
|---|---|---|---|
| Base | 0.646 | 0.590 | 0.647 (P 0.65, S 0.64, FP 0.82) |
| Arm 2 cubic | 0.573, paired -0.018 [-0.079; +0.043] | 0.606, +0.043 [-0.014; +0.097] | 0.611, **-0.024 [-0.086; +0.029]** (P 0.57, S 0.68, FP 1.25) |
| Arm 1 synthesis, first version | 0.467, -0.154 [-0.287; -0.038] | 0.374, -0.206 [-0.274; -0.128] | 0.473, **-0.130 [-0.212; -0.056]** (P 0.55, S 0.49) |
Secondary measures (ensembles, `missed_ens2.txt`): sensitivity small (< 7.2 mm3) base 0.50 / cubic **0.68** / synthesis 0.36; low-contrast (below median) 0.46 / **0.62** / 0.33; large 0.74 / 0.87 / 0.72. False positives per case per cohort (raw): base 1.25 / 1.59 / 0.95, cubic 2.50 / 3.15 / 2.91, synthesis 2.25 / 1.85 / 3.86.
**Arm 1 (first version): NEGATIVE, confirmed, both seeds and ensemble.** Diagnosis and second version above (5ae-1b, [213]-[214]).
**Arm 2 (cubic): NO GAIN per rule (i) (s42 below base) and (ii) (ensemble interval around zero, point value -0.024).** Notable, but pre-classified as "negative" pattern: cubic interpolation makes the network MORE SENSITIVE (small bleeds 0.50 -> 0.68, low-contrast 0.46 -> 0.62, overall S 0.64 -> 0.68) and pays with false positives (0.82 -> 1.25 per case, doubled in cohort 2) -- exactly the pattern of blob/center (5j/5k): buy sensitivity, lose F1. The network wants higher thresholds (0.5-0.7); even at its own cross-selected cell it is still not better (0.623 against 0.647). Not adopted. Cubic remains an option should a classifier ever be able to separate the additional candidates (candidate ceiling 5w) -- today it cannot (5y).
**Arm 3 (Momeni/CSIRO, 57 SWI, 146 point references, sphere label 1.5 mm, without CSF rule, `momeni_*.txt`):**
| Model (zero-shot) | pooled F1 | P | S | FP per case | Median F1 per subject (challenge rule, 5 mm) |
|---|---|---|---|---|---|
| catalina-SWI s42 | **0.491** | 0.36 | 0.79 | 3.63 | 0.50 |
| catalina-SWI s1 | 0.407 (paired -0.084 [-0.108; -0.062]) | 0.28 | 0.77 | 5.19 | 0.40 |
| catalina-SWI ensemble s42+s1 | 0.447 | 0.32 | 0.77 | 4.26 | 0.50 |
| **VALDO-T2* s42** | **0.525** (paired against SWI s42 +0.034 [-0.011; +0.077]) | 0.39 | 0.81 | 3.30 | 0.50 |
Expectation checked: SWI model 0.30-0.55 -- met (0.49). "VALDO model clearly below" -- **WRONG**: the VALDO-T2* model is on this foreign SWI cohort on par with to better than the own SWI model (on catalina-SWI it was 0.305 against 0.626). Read: (1) both models find ~80% of the marked bleeds without any adaptation -- the first INDEPENDENT public number of the line, and it is at VALDO level in sensitivity; (2) the gap sits in precision (0.36-0.39, 3.3-3.6 false positives per case), and there are three suspects for that, which would correct the number upward: no CSF rule (on VALDO +0.06), point reference only of the "definite" bleeds (possible/uncertain ones are not marked and count as false positives), and 1.5-mm spheres instead of masks (touch rule stricter than with real masks); (3) the SWI ensemble is WORSE than its best seed (0.447 against 0.491), because s1 drops noticeably here -- the two-seed rule (5p: "never ship the best seed") still holds, because which seed leads on a foreign cohort is not known beforehand (on catalina-T2* it was s1); (4) the VALDO model transfers better to 1.75-mm SWI than to the 2-mm catalina-SWI -- the Momeni images are histogram-matched and skull-free like VALDO, catalina-SWI has the weakest contrast (5n); the cohort difference is thus not "SWI vs. T2*", but image statistics. Limits: time points of the same subject count as cases (intervals too narrow); no CSF rule (SynthSeg on 57 cases would be ~2 CPU hours with `--parallel 2`); reference = points.
**Conclusion for Arm 3:** as external validation into the paper (sensitivity ~0.8, F1 ~0.5 without adaptation), with the three caveats; follow up with SynthSeg + CSF rule as soon as the GPU queue is empty (avoid CPU load next to training).
**Followed up 29 Sep 00:10 -- SynthSeg --robust on the 57 Momeni cases (`cmb/transfer/momeni_synthseg.py`, 22 s per case, CPU; maps on the grid under `<extern>/synthseg/`) and CSF rule (`momeni_op_csf.json`, `momeni_subject_csf.json`):**
| Model (zero-shot, fixed cell + CSF rule) | F1 pooled | P | S | FP per case | Median F1 per subject | without rule (above) |
|---|---|---|---|---|---|---|
| catalina-SWI s42 | **0.555** | 0.49 | 0.64 | 1.68 | 0.667 | 0.491 |
| catalina-SWI s1 | 0.509 | 0.42 | 0.64 | 2.26 | 0.533 | 0.407 |
| catalina-SWI ensemble | 0.522 (cross-selected 0.570) | 0.44 | 0.64 | 2.07 | 0.571 | 0.447 |
| **VALDO-T2* s42** | **0.613** (paired against SWI s42 +0.028 [-0.031; +0.090]) | 0.58 | 0.65 | 1.21 | **0.667** | 0.525 |
The CSF rule raises all four models here by 0.06-0.10, as on VALDO -- but unlike there NOT for free: sensitivity falls from ~0.80 to ~0.64, so the rule deletes about a fifth of the marked bleeds (on VALDO 0 of 139). Presumably more marked bleeds sit at sulci in Momeni (point reference, 1.75-mm slices, tighter skull-free mask: 36% of the image voxels carry no SynthSeg label). For Momeni the SOFT version (joint CNN with tissue feature, 5e) would be the right way, but it is not available without an application mode (Arm 4 builds exactly that).
**Balance Arm 3, for the paper:** external validation on a foreign public SWI cohort without any adaptation -- median F1 per subject 0.667, pooled 0.55-0.61 with CSF rule resp. 0.49-0.53 without (sensitivity 0.8), the VALDO model on par with or better than the SWI model. Caveats: point reference of only the certain bleeds, sphere label, time points of the same subject, CSF rule costs sensitivity here.
**RESULT ARM 4 (29 Sep 06:50; `mb-arena/stage2/transfer_valdo_{0,1,2}.json`, three seed offsets, mean; difference to the joint CNN of the same run):**
| Classifier | catalina-T2* F1 (difference) | catalina-SWI F1 (difference) |
|---|---|---|
| threshold only | 0.654 (-0.016) | 0.626 (-0.046) |
| threshold + CSF rule | 0.670 (0.000) | 0.614 (-0.058) |
| joint CNN (pool of all cohorts) | **0.670** | **0.672** |
| VALDO classifier, zero-shot (fraction 0) | 0.648 (-0.022; P 0.75, S 0.57) | **0.671 (0.000)** |
| fine-tuned with 10% of the training cases | 0.606 (-0.064; one offset 0.65, two 0.58) | 0.620 (-0.051) |
| 25% | 0.644 (-0.025) | 0.651 (-0.021) |
| 50% | 0.666 (-0.004) | 0.662 (-0.009) |
| 100% | 0.670 (0.000) | 0.660 (-0.012) |
**Read against the expectations:** (1) "fraction 1 within ±0.01 of the joint CNN" -- met (T2* 0.000, SWI -0.012 just missed). (2) "zero-shot on SWI clearly below" -- **WRONG**: the pure VALDO classifier is on SWI ON PAR with the joint CNN (0.671 against 0.672, within 0.008 in all three offsets), on T2* only 0.022 below -- and there with HIGHER precision (0.75) at lower sensitivity: the VALDO threshold is too strict for T2*, the classifier itself transfers. (3) "25% suffices on T2*, 50% on SWI" -- half right: T2* needs 50%; on SWI fine-tuning brings nothing at all, and with 10-25% of the cases it HURTS (overfitting to few cases plus unstable threshold from few inner candidates; one offset drops to 0.562). Side finding: the SWI gain of the joint CNN against threshold + CSF rule is there again with three further offsets (+0.058), the seed spread of the CNN lies at 0.003-0.013.
**Conclusion that counts:** the PUBLISHABLE chain (U-Net + classifier trained only on VALDO, without any private candidate) is on both catalina cohorts as good as today's private pool chain (SWI equal, T2* -0.022 with too-strict threshold). That is the chain that belongs in the public repo; fine-tuning on local candidates only pays off from about half the cohort onward and then brings at most 0.01-0.02. For Momeni (Arm 3) it follows that the VALDO classifier can be applied there as a soft tissue alternative to the hard CSF rule -- an application mode (train classifier on all VALDO candidates, apply to foreign candidates) is now trivial to build with `fit_predict_cnn_transfer` (fraction 0). Next step, if wanted.
**RESULT ARM 1, SECOND VERSION (5ae-1b; 29 Sep 06:50; `synv2_op_*.json`, `synv2_missed_ens2.json`):** fixed cell + CSF rule s42 0.590 against 0.646 (cross-paired -0.024 [-0.088; +0.044]), s1 0.532 against 0.590 (-0.041 [-0.105; +0.021]), ensemble 0.605 against 0.647 (**-0.072 [-0.129; -0.018]**, P 0.55 S 0.60 FP 1.21 against P 0.65 S 0.64 FP 0.82). Secondary measures as expected: sensitivity small 0.50 -> 0.57, low-contrast 0.46 -> 0.51, large 0.74 -> 0.83 -- but paid with false positives. **Rule (i) and (ii) violated: negative.** The second version confirmed the diagnosis (budget preserved -> no more collapse as in version 1, -0.13 -> -0.04 at the raw ensemble) and then shows the same pattern as every training lever of this line (5j/5k, cubic 5ae): more sensitivity, less F1. **Arm 1 is closed.** Balance: artificial bleeds in the first stage give the network no information that separates bleed from confounder; they only lower the bar. CenSynCMB's gain came together with T1 + T2 + T2* and labeled confounders, which taught us nothing -- our finding does not contradict theirs, it shows the condition.
**Arm 5 -- vessels as false positives: dark running track along z (user observation 29 Sep in the slice viewer: "many false positives are vessels that run over several slices; bleeds are not that long in z"; fixed in advance 29 Sep ~08:50).**
Why this could be new: FRST is 2D per slice and cannot separate vessel cross-section from bleed; the feature "extent through the slices" of the feature models only measured the component AFTER the threshold (a vessel breaks into pieces at 0.15); the two-scale classifier lowered false positives on T2* (3.0 -> 2.7), so there is something to gain there; inpainting (original) deleted bleeds along with it and is not a way forward.
Measurement (`cmb/analysis/vessel_run_length.py`, CPU): candidates = components of the out-of-fold map at 0.15 / >= 1 mm3 (as in the second stage), per candidate: (a) z-extent of the component itself, (b) **dark running length in the IMAGE**: from the candidate center, slice by slice, follow the local brightness maximum of the inverted scale (window ±1 mm in-slice for oblique vessels), as long as the contrast to the ring (1-2 mm) holds at least half the contrast at the center; length in mm upward plus downward plus 1. Real bleeds have a median z-extent of 3 slices on the grid, p90 6 (cohort1 4-8 due to block-filling). Evaluated on VALDO (57), catalina-T2* (75), catalina-SWI (61), Momeni (57): distribution of the running length for real vs. false candidates (percentiles, AUC), then the rule "running length > L mm -> discard" with L from {6, 8, 10, 12, 15, 20} chosen per fold on the OTHER folds, paired against the fixed cell and against fixed cell + CSF rule.
**Expectation, in advance:** the running length separates real from false candidates with AUC 0.65-0.75 (not all false positives are vessels; 5i: many sit in sulci/CSF, which the rule already catches); as a rule +0.01 to +0.03 F1 on T2*/SWI (5-mm resp. 2-mm slices: there vessels run through few slices, the effect is limited) and +0.00 to +0.02 on VALDO; TP loss under 3%. Gain counts if it lies paired above zero AND discards fewer than 3% of the reference bleeds. If the AUC falls below 0.6, the observation is qualitatively correct but not the main share of the false positives, and step 3 (elongated classifier patch) is measured anyway, because the classifier sees more than a running length.
**Result step 1+2 (29 Sep ~09:10; `ergebnisse/analysis/5ae_0929/vessel_*.json`; candidates of the fixed cell; rule chosen per fold on the other folds):**
| Cohort (first stage) | real / false candidates | running length real p50 / p90 | false p50 / p90 | AUC (false longer) | rule on fixed cell | rule on cell + CSF | TP loss |
|---|---|---|---|---|---|---|---|
| VALDO (ensemble) | 95 / 95 | 5 / 15 mm | 6 / 21 mm | 0.55 | -0.021 [-0.056; +0.014] | -0.009 [-0.031; +0.015] | 11% / 6% |
| catalina-T2* (own) | 562 / 309 | 11 / 23 mm | 11 / 23 mm | **0.47** (real tend to be longer) | **-0.047 confirmed** | -0.054 confirmed | 13% |
| catalina-SWI (own) | 435 / 290 | 7 / 16 mm | 7 / 30 mm | 0.55 | -0.004 n.s. | -0.015 n.s. | 7% |
| Momeni (VALDO U-Net) | 123 / 224 | 5 / 8 mm | 9 / 28 mm | **0.72** | **+0.108 [+0.078; +0.142]** (0.499 -> 0.607) | **+0.044 [+0.022; +0.071]** (0.601 -> 0.645), L = 10 mm | 6% / **2%** |
z-extent of the component itself: AUC 0.27-0.49 everywhere, real ones are LONGER there (block-filling of thick slices) -- unusable as a feature, as suspected.
**Read:** the observation is quantitatively correct on Momeni and holds up: false candidates run there median 9 mm, p90 28 mm, real ones 5 / 8 mm; the rule "discard longer than 10 mm" raises F1 even AFTER the CSF rule, confirmed, by 0.044 at 2% TP loss -- per the pre-fixed rule a gain, and 0.645 is the best Momeni number (above the VALDO classifier's 0.607). On VALDO and catalina, in contrast, the running length does not separate (AUC 0.47-0.55), and on T2* the REAL candidates tend to be longer: at 5-mm slices and pronounced blooming, any bleed reaches over 2-3 slices = 10-15 mm, and the tracing runs through. Also on VALDO, 10 of the 23 long real candidates come from cohort 1 (4 mm). Of the 27 long false VALDO candidates, 9 sit in CSF -- the CSF rule already hits part of the same vessels.
**Decision per pre-fixed rule:** rule NOT into the general chain (neutral to confirmed worse on three of four cohorts). For thin-slice, high-contrast SWI like Momeni it is effective; it is kept as an optional switch (`vessel_run_length.py`, L from the inner folds of the target cohort). The expectation "AUC 0.65-0.75 everywhere" was too optimistic; it only held for Momeni. The simple tracer is not the general solution -- **step 3 (classifier with long z patch) is measured**, because it can learn the pattern that the hand-made rule misses at thick slices.
**Step 3, fixed in advance (29 Sep ~09:20):** candidates with `--half 20 20 20` (patch 20 x 20 x 40 mm, folder `-weich-z40`), classifier `cnn-z32` = the same CNN with a patch of 32 mm along z and 16 mm in-slice (`fit_predict_cnn(crop=(32, 32, 32))`; network is fully convolutional, otherwise unchanged), paired against `cnn` (16-mm cube) on the SAME candidates in the same run, three seed offsets. Expectation: T2* +0.01 to +0.02 (there the two-scale context also helped), SWI ±0.01, VALDO ±0.01; false positives per case fall, sensitivity stays. Gain counts if the mean over three offsets lies above +0.01 AND VALDO does not fall below -0.01 (rule from 5r). Afterward, if positive, `apply.py` with the same patch on Momeni.
**RESULT step 3 (29 Sep 12:55; `mb-arena/stage2/z32_{0,1,2}.json`, three offsets, `cnn-z32` against `cnn` on the same z40 candidates):**
| Cohort | CNN 16-mm cube (mean) | CNN 32 mm z context (mean) | difference per offset | mean | P / S / FP per case |
|---|---|---|---|---|---|
| VALDO | 0.630 | 0.616 | +0.001 / -0.025 / -0.018 | **-0.014** | 0.65 -> 0.64 / 0.61 -> 0.59 / 0.79 -> 0.81 |
| catalina-T2* | 0.669 | 0.681 | +0.007 / +0.011 / +0.017 | **+0.012** | 0.70 -> 0.71 / 0.64 -> 0.66 / 3.08 -> 3.03 |
| catalina-SWI | 0.672 | 0.669 | -0.001 / +0.004 / -0.011 | -0.003 | 0.73 -> 0.71 / 0.62 -> 0.63 / 2.41 -> 2.74 |
**Read against the rule:** T2* meets "> +0.01 on average" (all three offsets positive; the seed spread of the second stage lies at 0.003, so the effect is real), but VALDO falls with -0.014 below the -0.01 limit -> **not adopted into the general chain.** Same pattern as with the two-scale CNN (5r: T2* +0.016/+0.019, VALDO -0.030): on 5-mm T2* more context along z helps the classifier a little (vessel continuation, blooming across slices), nothing on SWI, and on VALDO with only 278 candidates the larger input costs. For a pure T2* release, `cnn-z32` is the best measured second stage (0.681 against 0.669 -- and the two-scale CNN was still somewhat higher at 0.687-0.691). Arm 5 thereby concluded: the vessel question cannot be moved significantly on thick slices, neither with rules nor with more z context; the rest is the physics of the acquisition.
**Step 2b -- Frangi vesselness and Hessian blobness as post-processing (user idea 29 Sep ~09:30; `vessel_run_length.py` extended, `ergebnisse/analysis/5ae_0929/frangi_*.json`; per candidate on a local, isotropically 0.5 mm resampled patch, scales 0.5-1.5 mm, maximum in the 1 mm expanded candidate):**
| Cohort | Frangi AUC (false more vessel-like) | Blobness AUC (real more blob-like) | rule vesselness (nested) | rule blobness | rule blobness + running length |
|---|---|---|---|---|---|
| VALDO | **0.42** (real are more vessel-like!) | 0.62 | selected out, ±0 | -0.010 | -0.010 |
| catalina-T2* | **0.45** | 0.58 | out, ±0 | out, ±0 | out, ±0 |
| catalina-SWI | 0.62 | 0.48 | out, ±0 | out, ±0 | out, ±0 |
| Momeni | **0.25** | **0.82** | out, ±0 | out, ±0 | = running length alone, +0.044 |
**Read:** Frangi inverts -- the REAL bleeds are more vessel-like by Frangi than the false positives (Momeni AUC 0.25, VALDO 0.42, T2* 0.45). The reason lies in the grid: a bleed with blooming on 2-5-mm slices becomes, after resampling to 1 mm, a short cylinder along z, and exactly that is what a tube detector fires on; the fine scales (0.5-1.5 mm) also see the edge of a 3-mm bleed as a tube. Blobness separates well on Momeni (AUC 0.82), but as a hard rule after the CSF rule it brings nothing anywhere (the remaining false positives there are similarly blob-like as the real ones); on VALDO it costs. **Decision: Frangi/blobness not as a rule.** As a feature for the classifier, blobness would be conceivable on thin-slice data, but the CNN already sees the shape itself -- step 3 measures exactly that with more z context. Tool lesson: every hand-made shape filter must be measured on THIS grid (anisotropically resampled, blooming); the intuition from the isotropic 7T image does not transfer.
**Step 2c -- border rule per Sundaresan 2023 (discard candidates < 5 mm from the brain mask edge; measured 29 Sep midday, `border_*.json`, distance via EDT on the grid, d from {2, 3, 4, 5, 7 mm} nested per fold):** AUC "false closer to the edge" 0.49 / 0.49 / 0.51 / 0.48 (VALDO / T2* / SWI / Momeni) -- false and real candidates sit equally far from the edge (median 28 / 17 / 13 / 27 mm). Rule after CSF rule: +0.002 / -0.002 / 0.000 / 0.000, everywhere without meaning. Reason: for us the sulcus share is already covered by the CSF rule (SynthSeg), and the brain masks (VALDO head mask, HD-BET) run differently from OXVASC's; the remaining false positives sit deep in the parenchyma. **Not adopted.** With that, all three of Sundaresan's post-processing rules have been checked on our data: volume (we have it), ellipticity (flips, Frangi finding), edge (ineffective). The precision of 0.84 on OXVASC arises there from the combination with inpainting and distillation on 5-mm data in native resolution -- a setting we can only reproduce with the fine-tuned original itself (Arm 6, running).
**Step 2d -- object geometry of the candidate component (user question 29 Sep afternoon after light post-processing; literature `research/light-postprocessing-2026-09-29.md`: axis ratio > 3 = vessel in the geometric cascade 2026, object features in van den Heuvel 2016; measured `geom_*.json`):** per candidate PCA axis ratio (longest/shortest principal axis in mm), in-plane ratio per slice, centroid shift between neighboring slices, number of occupied source-slice blocks; rules nested after the CSF rule.
| Cohort | AUC axis ratio (false more elongated) | AUC in-plane | AUC shift | rule axis ratio | rule in-plane | rule shift |
|---|---|---|---|---|---|---|
| VALDO | 0.54 | 0.59 | 0.47 | -0.003 | -0.001 | -0.005 |
| catalina-T2* | 0.56 | 0.61 | 0.53 | +0.000 | +0.001 | 0.000 |
| catalina-SWI | 0.49 | 0.62 | 0.54 | +0.002 | -0.001 | +0.001 |
| Momeni | 0.67 | 0.57 | 0.67 | -0.002 | -0.007 | 0.000 |
Number of occupied blocks: AUC 0.34-0.51, real ones occupy MORE (block-filling), as already the z-extent. **None of it carries** -- all rules ±0.005, even on Momeni, where the running length in the image brought +0.044. Explanation: the component is that of the U-Net at threshold 0.15, not the structure in the image; a vessel that the network only marks pointwise is as a component just as compact as a bleed (median axis ratio real 1.8-2.7 against false 1.9-2.9). With that, all "light" shape levers from the literature have been measured on our data: running length (Momeni only), Frangi, blobness, edge, axis ratio, in-plane shape, shift, block count. **Step 2 is closed; for vessel false positives there is no rule on this grid, only the learned network (CSF rule and classifier).**

**Balance of research line 5ae (four arms, ~13 GPU hours):** no arm raises VALDO F1; two expectations turned out wrong, both in the same direction: **the VALDO models transfer better than thought** (T2* U-Net on Momeni-SWI 0.613 against SWI U-Net 0.555; VALDO classifier zero-shot on SWI on par with the pool). That is the result that belongs in the paper: a purely public, publishable chain, externally validated.
**Application mode built and applied to Momeni (29 Sep 07:40, user approval; `cmb/stage2/apply.py`, `ergebnisse/analysis/5ae_0929/momeni_apply_valdo_classifier.json`):** classifier (same CNN, three seeds x three members) trained on ALL 278 VALDO candidates, threshold 0.50 from VALDO's own inner folds (out-of-fold F1 there 0.620), nothing from Momeni used for any choice. Momeni candidates with `cmb.stage2.candidates --keep-csf` from both zero-shot runs (534 resp. 529 candidates, sensitivity ceiling 0.88 / 0.85).
| first stage on Momeni | fixed cell (0.3 / 2 mm3) | + hard CSF rule | + VALDO classifier | paired classifier against rule |
|---|---|---|---|---|
| VALDO-T2* U-Net s42 | 0.499 (P 0.35, S 0.84) | 0.610 (P 0.55, S 0.69) | 0.607 (P 0.54, S 0.70) | -0.003 [-0.056; +0.050] |
| catalina-SWI U-Net s42 | 0.455 (P 0.32, S 0.80) | 0.524 (P 0.43, S 0.67) | **0.630 (P 0.59, S 0.67)** | **+0.106 [+0.044; +0.159]** |
**Read:** on the candidates of the VALDO U-Net the learned classifier is on par with the hard rule (as on VALDO itself, 5e); on the candidates of the SWI U-Net it beats the hard rule, confirmed, by 0.106 and delivers with 0.630 the best Momeni number of the line -- without a single Momeni or catalina candidate ever going into training. It costs sensitivity like the rule (0.80 -> 0.67; the lost bleeds are presumably the same sulcus-near points), but with clearly better precision. With that the purely public chain is complete and externally validated: VALDO U-Net -> VALDO classifier, F1 0.61 / median per subject 0.667 on a foreign SWI cohort. For deployment, `apply.py` is the mode; `accepted` in the JSON holds the accepted candidates per case ready for a mask output.

**User preference 29 Sep 10:50:** "the model with the highest sensitivity I like best" -- for deployment, sensitivity counts before precision (pre-sorting for the rater). That is today the cubic ensemble with CSF rule (sensitivity 0.68 against 0.64, small bleeds 0.68 against 0.50, false positives 1.25 instead of 0.82 per case; F1 0.611 against 0.647) resp. on catalina-SWI blob (small 0.44 against 0.37). Consequence: run a "sensitive chain" alongside the F1 chain and report both. User task at the same time: measure our chain against the FINE-TUNED original (MicrobleedNet, pretrained weights + fine_tune) on VALDO and catalina (Arm 6, see below) and prepare the core experiments as a repeatable Python script.

**Arm 6 -- fine-tuned original as a fair yardstick (user task 29 Sep 10:55).** `cmb/baselines/microbleednet_finetune.py`: the original (pretrained weights, own preprocessing `data_preparation.preprocess_subject`, `fine_tune` with all layers, 60 ep., early stop 10, `evaluate`) on OUR folds; evaluation native (mb_metrik logic) and on our grid (prediction resampled nearest-neighbour, `cmb.transfer.evaluate`). Available from the arena (09/2026, preprocessing preproc2, all layers): catalina-SWI -- on our grid with our metric **0.406 (P 0.34, S 0.51, 10.6 FP per case) against our U-Net 0.634, paired -0.228 [-0.308; -0.165]**; viewer `mb-arena/qc/swi_vs_sundaresan/`. Queued: T2* five folds [218]-[222] (~75 min per fold); VALDO follows as soon as preprocessing is through. Expectation in advance: original fine-tuned on VALDO 0.30-0.45, on T2* 0.40-0.55 -- clearly below our chain, but above the rebuild without fine-tuning (0.245).
**Repeatability:** `cmb/experiments/kern.py` runs the core steps as one script (`--list`, `--dry-run`, `--queue`); run of the CPU steps 29 Sep 11:05 in `logs/kern_wiederholbarkeit_0929.log`.

**Arm 7 -- epoch dose 90 for the paper (user decision 29 Sep 12:40, fixed in advance):** queued behind Arm 6: catalina-T2* own training 90 ep. (seeds 42 and 1), fine-tuning from the VALDO model 90 ep. (lr 1e-4), VALDO 90 ep. (seed 42). Yields the dose series 60 / 90 / 120 for training and 10 / 30 / 90 for fine-tuning. **Expectation:** 90 lies between 60 and 120, i.e. -0.01 to -0.02 against 60 (T2*: 0.663 / ? / 0.640; VALDO: 0.585 / ? / 0.556), fine-tuning 90 equal to 30 (curve flat from epoch 16). Rule: no adoption below +0.03 with a single seed; the series is shown in the paper as a curve, not as a decision.

**Arm 8 -- T1 and T2 as additional channels, VALDO only (user task 29 Sep 13:30; fixed in advance):** ceiling of the single-sequence solution (5q e). The three challenge leaders and CenSynCMB use T1 + T2 + T2*. `code/zusatzkanaele_bauen.py` lays VALDO's masked T1/T2 (T2S space; 27 of the 57 CV T1 carry NaN -> nan_to_num) linearly onto `gitter_d`, divides in the brain by the 99.5th percentile, caps to [0, 1], not inverted (mean in the brain T1 0.41, T2 0.47). `turnier.py --zusatz t1 t2` (backup `.vor-zusatz-0929`): channels 3 and 4 next to T2* and FRST, network with 4 inputs (22.50M parameters instead of 22.50 -- only the first convolution grows), sample word-identical to the base (test: shape (8, 4, 38, 62, 94), channel mean 0.41 / 0.03 / 0.33 / 0.31; without the switch everything stays at 2 channels). Runs: seeds 42 and 1, 60 epochs, paired against the base with CSF rule (0.646 / 0.590; ensemble 0.647). **Expectation:** +0.02 to +0.05 on VALDO, mainly in precision (T1/T2 mark sulci, vessels and calcification differently from bleeds); rule as always (both seeds above base, ensemble interval above zero). Not transferable to catalina (no T1) -- the number only quantifies how much of the gap to CenSynCMB (74.3) lies in the sequences.

**Arm 9 -- tournament 2: hyperparameters and further networks, 60 epochs (user task 29 Sep 14:00 "extensive hyperparameter search"; fixed in advance).** Fixed base a03-aniso, `gitter_d`, 60 ep., seed 42, interim-state selection; exactly one change per run. Reference: base s42 0.585 raw / 0.646 with CSF rule (cross-selected 0.622). 17 runs, ~1.6 h each (b48, n_patches 1600 longer), queued behind Arm 6/7/8:
| Group | Runs (folder suffix) | Reference in code |
|---|---|---|
| Width | `a03-aniso-b16` (5.6M), `a03-aniso-b48` (50.6M) | Base 32 (22.5M) |
| New networks | `a13-r2unet2p5d` (recurrent-residual 2D U-Net on 5 neighbouring slices, 8.9M), `a14-mednext` (MedNeXt-S, MONAI 1.5, 5.6M), `a15-nnunet-resenc` (DynUNet with residual encoder, 16.7M) | Tournament 1: a04 2.5D 0.294, a02 nnU-Net 0.458 (e60, clean) |
| Learning rate | `-hplr0.0001`, `-hplr0.001` | 3e-4 |
| Weight decay | `-hpweightdecay0`, `-hpweightdecay0.001` | 1e-4 |
| Batch | `-hpbs2`, `-hpbs8` | 4 |
| Budget per epoch | `-hpnpatches400`, `-hpnpatches1600` | 800 (5h: budget per case is the bottleneck) |
| Lesion fraction | `-l40`, `-l80` | 60 / 25 / 15 |
| Loss | `-vfocal_tversky` (alpha 0.3, beta 0.7), `-vbce_dice_ohne_ds` | BCE + Dice with deep supervision |
Code: `code/netze.py` (backup `.vor-turnier2-0929`; forward passes of all five new networks on (2, 38, 62, 94) checked, ResEnc delivers three heads in training like a02), `turnier.py --hp SCHLUESSEL=WERT`, `--verlust focal_tversky | bce_dice_ohne_ds`.
**Evaluation and rule, in advance:** all runs together with `operating_point` (fixed cell + CSF rule, cross-selected as secondary value) paired against the base s42, plus raw per count class. With ONE seed nothing below 0.08 is interpretable (Section 3): runs with ≥ +0.04 against the base get a second seed; only what lies above the base with both seeds and has an interval above zero in the two-seed ensemble is adopted. Expectation: no change above +0.04 (all previous training levers were at 0 or below); b48 and MedNeXt most likely on par, 2D recurrent clearly below (~0.3 like a04), n_patches 1600 slightly worse (budget), lr 1e-3 unstable. The tournament serves the paper as evidence of the limited search, not as hope for a leap.

**Arm 9, addendum (user decision 29 Sep 13:50, per `research/model-selection-practice-2026-09-29.md`):** second seeds sampled also for non-winners -- two tournament runs drawn at random (Python `random.seed(20260929)`, `random.sample`): **`n_patches=1600` and `bs=2`**, each `--seed 1`, queued behind the tournament. Purpose: the threshold "+0.04 -> second seed" otherwise only applies to winners; here it is checked whether an unremarkable run stays equally unremarkable with a different seed (Picard 2021).
**Arm 10 -- real nnU-Net v2 as baseline (user decision 29 Sep 13:50; "nnU-Net Revisited": standardized baseline missing):** nnunetv2 2.8.1 (dl environment), dataset `Dataset501_VALDOCMB` from the 57 CV cases -- input the masked VALDO T2S in native resolution (no FRST, none of our preprocessing; nnU-Net plans resampling and normalization itself), label CMB, our five folds as `splits_final.json`; configuration 3d_fullres, trainer `nnUNetTrainer_250epochs` (default would be 1000; 250 for budget reasons, disclosed), 5 folds; prediction out-of-fold, evaluation with our lesion metric native and resampled to the grid (like Arm 6). **Expectation:** 0.45-0.60 with CSF rule (the DynUNet rebuild was at 0.458 raw; the framework brings augmentation, ensemble-free, own normalization), below our chain (0.646). Tool `cmb/baselines/nnunet_v2.py`.
## 6 Threats to validity
- 57 cases, 139 CMB: wide intervals; many comparisons on the same folds -> risk of overfitting to the CV.
- The CSF rule and inpainting 2 were designed after looking at the CV data (inpainting parameters without label).
- The 15 held-out cases were used once on 14 Sep (F1 0.19, one case carries 75% of the CMB).
- Reference annotation from three groups with partly different protocols; inter-rater value not reported in the paper.
- Independent confirmation is still pending: catalina (different scanner, weaker contrast, 5 mm slices).
- **28 Sep:** the 15 held-out cases were measured for the second and last time (5ad): pooled 0.468 (one case with 73 bleeds, S 0.20), without it 0.731, median per case 0.667. Both numbers belong in the paper; no choice was made after this look.

## 7 Next steps
**Status 28 Sep (after 5ac):** training recipe (5h, 5o, 5p, 5t, 5v), intensity scale (5aa, 5ac), architecture (5s), candidate ceiling (5w, 5y), second stage (5r, 5z, 5ab) have been measured; no lever is confirmed above the baseline. **The chain is finished:** per cohort base U-Net (two-seed ensemble) + shared small 3D-CNN + CSF rule on VALDO. What matters now is no longer measuring but concluding: (1) the expert judgment on the blinded review page (5b, with the user since 22 Sep) -- it decides how much of the residual error is reference; (2) the ONE-TIME measurement of the frozen chain on the 15 held-out cases and by challenge metric; (3) paper numbers as the mean over seeds (5u), limits from section 6; (4) publication (we6/we7, user decision). The only still-unmeasured lever with DIFFERENT information (rule from 5q): T1/T2 as additional channels, as with the three challenge best -- one training run, if the user wants it, but only after (1).
**Status 21 Sep 21:00:** the original plan below (from 19 Sep, left in place for traceability) has been worked through except for large patches and long training (queued) and the one-time measurement on the 15 held-out
cases (deliberately last). The current plan -- what has been exhausted, what is running, what remains open -- is in 5q.

Inpainting 2 vs. none -> second seed of the new base (gitter_d) -> bias off / percentile scale / z = 2 mm /
without FRST / ratio / attention / large patches / 300 epochs, all on gitter_d -> confirm the best recipe with a second
seed -> two-stage version -> tissue map GM/WM/CSF/CMB -> catalina (without adaptation / fine-tuned /
catalina-only as control) -> one-time measurement on the 15 held-out cases.

## 8 Code
The experiment code (`code/`, German, grown in the course of exploration) is being replaced module by module by the English
package `cmb/` (user request 19 Sep: readable, structured, easy to debug). Acceptance rule: a module counts only once
it reproduces its predecessor exactly. Status 19 Sep: network building blocks and the winning network ported; with the saved
tournament weights the outputs are bit-identical to the old code (`python -m cmb.tests.test_models_match_legacy`).
Plan and status per module: `cmb/README.md`.
**New 29 Sep:** `cmb/review/build_slice_viewer.py` -- slice viewer without a server: two large axial images per case (left reference, right any number of model outlines as toggleable colors), mouse wheel/arrows = z-axis, both sides synchronized, TP/FP/FN counter per model and per slice; user request. Built for VALDO (`ergebnisse/qc/valdo_slices/betrachter.html`, models basis / basis+liquor / kubisch / synthese2); callable for catalina with private `--out` and the `mb-arena` runs.
Addendum 29 Sep 09:30: the page is responsive (phone: case selection on top, swiping over the image = slice, buttons at the bottom; images stacked in portrait orientation) and has `--single-file N` (single file with the N cases richest in bleeds embedded, 480 px, JPEG 72; 24 cases = 29.6 MB). The single file with four models was sent to the user via `schicken` (Telegram) -- VALDO only, public. Second viewer with 31 VALDO runs: `ergebnisse/qc/valdo_slices_alle/` (tournament runs on `gitter_c` do not fit on the image and are missing).

## 9 Teaching
`cmb/teaching/build_exercise.py` generates an exercise page for the lecture (one HTML file, offline, German/English):
students mark microbleeds in 12 T2* slices (three public VALDO cases, easy/medium/hard), get three attempts per
case and receive Dice, F1, sensitivity, precision and specificity back; own marks are colored green
(hit) or red (false positive), the missed bleeds appear yellow after the third attempt. Specificity is included
deliberately: it is close to 1.0 for everyone and shows why it says nothing for tiny lesions.
The scoring logic is tested against a hand example (Dice 0.444, F1 0.5). File: `ergebnisse/lehre/cmb_exercise.html`.

## 10 Traceability (status 2026-09-20) -- what is available for a publication and what is still missing
**Available:** (1) `protokoll.md`: every measurement with date, numbers, intervals; (2) this document: chain of reasoning per intervention,
pre-specified decision rules and expectations, negative results; (3) `DEVIATIONS-microbleednet.md`: differences from the
original with reference commit 958f1cb; (4) per training run and fold `meta.json` (all recipe parameters, seed, chosen epoch,
threshold, duration, VRAM) + `verlauf.json` + model weights + probability maps; (5) the queue file with every
training command, timestamp and return value; (6) backup copies of every changed script file with date (`dev/*.vor-*`);
(7) `docs-umgebung/`: frozen conda environments (dl, shiva), hardware, versions of PyTorch, MONAI, FreeSurfer, FSL;
(8) data provenance and licence (`dev/licence-valdo.md`), fixed split (`dev/split.json`, seed 0), determinism demonstrated
(a01 reproduces cv5-r3 to four decimal places); (9) since 20 Sep a private Git repository with code and documents.
**Gaps, honestly stated:** (a) Until 20 Sep there was NO version control; the code state per run can only be reconstructed via dated
backup copies. (b) A number of analyses ran as one-off scripts in the session and do NOT exist as a file:
paired bootstrap comparisons, cohort breakdown, damage assessment of the original inpainting (first version), mask volume,
contrast loss from resampling, FRST hit rate and AUC, error analysis, grid search threshold x minimum size, recalculation by
challenge rule. The NUMBERS are in the protocol. **Supplied afterwards on 20/21 Sep and checked against the protocol (table in `REPLICATION.md` section 6):** exactly reproduced are the
damage assessment (`code/uebermalung2.py --pruefen dev/preproc`), mask volume (`cmb/analysis/mask_volume.py`) and cohort breakdown (`cmb/analysis/cohort_breakdown.py`);
NOT exactly reproducible are the FRST hit rate (file: 82% -> 96% instead of 85% -> 96%) and the contrast loss (file: -26 / -15 / -30% instead of -23 / -12 / -32%) -- the
one-off code is lost, from now on the values from the files apply; the conclusions do not change. (c) Literature references in `research/` were extracted from the sources by a small reading model and must be checked line by line against the sources before
citing. (d) Errors in the original (A1-A3) were read from the source code, not executed.
(e) The 15 held-out VALDO cases were used once on 14 Sep -- disclose in the paper. On 28 Sep for the second and last time, with the frozen chain (5ad; checksums of the ten model files in `ergebnisse/validierung/heldout_0928/FROZEN.sha256`).

## 11 Terminology and naming check (2026-09-21, user request after the "3D-ResNet" misnomer; continued 22 Sep)
Checked: every designation in documents, reports and docstrings against what the code does (`code/netze.py`, `attention_unet.py`, `frst.py`, `cv5.py`, `turnier.py`, `cmb/`), plus the provenance of the data and
masks. **Incorrectly or imprecisely named -- corrected:**
1. **"3D-ResNet" (second stage)** -- is a plain 3D-CNN without residual connections (216,081 parameters). Corrected in 5j, in the docstrings and in `LITERATURE.md`. Numbers unaffected.
2. **"60 fixed epochs"** -- what is fixed is the plan, not the checkpoint used: the prediction comes from the best checkpoint of the inner fold (checked
   every 5 epochs). Chosen epochs e.g. basis
   40 / 60 / 50 / 35 / 60, seed 1: 50 / 55 / **10** / 40 / 40, catalina-T2* 55 / 60 / 20 / 55 / 35, fine-tuning "30 ep." 12 / 24 / 18 / 22 / 8. No leak (the inner fold is never the test fold), but: (a) the
   choice on ~11 cases is itself a source of noise -- part of the seed spread (0.553 vs. 0.585) is probably due to this; (b) statements about "epochs" (5h) concern the
   PLAN LENGTH. Suggestion: a run with a truly fixed final checkpoint as a control (switch still missing).
3. **Brain masks catalina "from HD-BET"** -- 58 of 75 (T2*) and 61 of 61 (SWI) masks are files supplied with the source (`*_brainmask.nii.gz`; a script `make_brainmask_hdbet.py` is located there, so very likely
   also HD-BET, not checked by me); only the 17 missing T2* masks were actually computed with HD-BET.
**Correctly named, but easily misunderstood -- clarifications:**
- *a02-nnunet*: the nnU-Net ARCHITECTURE (MONAI `DynUNet`, deep supervision), NOT the self-configuring nnU-Net procedure with its own training recipe. *a06 / a08 / a09*: MONAI `BasicUNetPlusPlus`, `SwinUNETR`
  (aborted), `SegResNet`. *a05*: real residual blocks with squeeze-excitation. *a01 / a11*: additive attention gates as in Oktay et al. "Ten architectures" = nine single-stage networks + the two-stage version; a11 / a12 were added later.
- *FRST*: a real Fast Radial Symmetry Transform after Loy & Zelinsky, but 2D per axial slice (`code/frst.py`), not 3D.
- *Zero-shot*: the mean of the FIVE VALDO fold models -- all other strategies are single models per fold. Since an ensemble is +0.02 to +0.04 above the average single model (5j), the domain drop is
  more likely LARGER than the measured -0.041.
- *"Scale max"* (user question 23 Sep 11:01: "what about 0-1 normalization?") -- misleading: our scale IS already a 0-1 normalization per case. What is delivered is `1 - T2*/maximum in the brain`; the inversion turns the brightest T2* voxel into 0
  and the darkest into 1. Re-measured on five cases: minimum in the brain 1e-6 to 1.3e-5, maximum 0.987 to 1.000 -- a min-max normalization would be a factor of 1.000-1.013 and a shift of one millionth, i.e. the identity.
  In the measurement (5aa), `minmax` therefore delivers exactly the same four numbers as `max`. Going forward it will be called "min-max per case on the inverted scale"; "divided by the maximum" (as in the original) reads as if the lower range remained unused.
- *Contrast* has three definitions: relative darkening against a mm ring (`missed_lesions.py`, 5i / 5n), difference on the inverted scale (`resampling_contrast.py`, 4.8; damage assessment 4.1), and the feature of the second
  stage (difference against a voxel ring). Do not compare values from different definitions.
- *F1* without a qualifier = pooled lesion F1 (26-connectivity components, touching) at the fixed cell 0.3 / 2 mm3; alongside it: inner F1 and F1 per fold in the training log (own threshold per fold), nested F1 of the second stage,
  VALDO rule (median per subject), F1 averaged per subject (CenSynCMB). Only compare like with like.
- *confirmed* = the 95% bootstrap interval over cases excludes 0; NO correction for the multitude of comparisons. **Addendum 22 Sep:** this interval does not contain the TRAINING noise -- two runs of the same recipe with
  a different seed differ by up to 0.083 and are reported as "confirmed different" (5u). From 22 Sep onward: single-seed effects under ~0.08 are directional indications, not proof (section 3).
- *Basis* (user question 22 Sep 21:25) -- I have used the word in TWO meanings: (a) the **base recipe**, i.e. the method (a03-aniso on gitter_d, T2* + FRST, 60-epoch plan with checkpoint selection, 800 patches
  38 x 62 x 94, fixed post-processing + CSF rule); (b) the **base run s42** (`ergebnisse/turnier/a03-aniso-e60-gd`, seed 42, 0.585 / 0.646), which is the reference in all paired tables because it finished first.
  This is dangerous now that 5u has shown that s42 is the best of three seeds: "base = 0.646" reads as a property of the method, but is in fact the property of a run's luck of the draw. **Language convention from here on:** "base recipe"
  for the method (to be reported as 0.605 +- 0.034, the two-seed ensemble 0.647 is what is shipped), "base run s42" for the comparison partner. In tables s42 remains the reference -- pairing is compared pairwise --,
  but a gap under 0.08 against exactly this run is not a statement about the recipe.
- *hard negatives*: mining in the SAMPLING of the U-Net; in Dou et al. 2016, by contrast, the second stage learns on the false positives of the first -- related, not the same. *blob loss*: conference proceedings not checked (arXiv 2205.08209).
**No findings:** losses (BCE+Dice, blob, fnrw, center do what the names say; deviations from the paper are in the docstrings), CSF and tissue classes (FreeSurfer table), bias correction (`fast -B`), grid,
sampling ratios, AdamW / cosine / InstanceNorm, ensemble (mean of the maps), mirror TTA (eight views), nested evaluation of the second stage, public QC page and its README (none of the misnomers).

### 29 Sep, reproducibility package on GitHub
New private repository `mendeltem/cmb-valdo-chain` (working copy ~/data/work/cmb-valdo-chain, commit 1f1550f): cmb/, code/, catalina/ scripts, docs/ (this document, REPLICATION, OVERVIEW, THESIS-CORE, research notes, environment), dev/ JSONs (split, manifests, landmarks), env/ (conda file + pip lock, 185 packages), README with installation, paths, VALDO BIDS layout, catalina BIDS derivatives (HD-BET / FAST -B once; existing derivatives are skipped) and the reproduction commands. Not in the repository: image data, weights, private case identifiers (the safety grep before the commit was empty). Open: pull the absolute paths `/home/uchralt` into one configuration; weights as a release asset.
Addendum 29 Sep, evening: (1) `pipeline/00_selftest.py` ... `11_qc_viewer.py` in the repository, number = order, every script calls the steps of `cmb/experiments/kern.py` (new there: preprocess, preproc-roh, synthseg, qc-viewer); `--dry-run/--force/--queue/--list`. The self-test without data (imports, FRST on synthetic discs/lines, lesion metric, U-Net forward pass, dry run of all steps) finishes in 5 s. kern.py in valdo-t2s is the same file. (2) The VALDO slice viewer (baseline, baseline+csf, cubic, synthetic2, each with/without CSF rule) is public at https://mendeltem.github.io/valdo-cmb-qc/slices/ (134 MB PNG, VALDO only), linked from the QC start page and from the repository README; the README links back to the QC page. (3) Late evening: the whole repository was switched to English (documents translated and renamed: ARBEIT -> WORKLOG, REPLIKATION -> REPLICATION, UEBERSICHT -> OVERVIEW-tests, MASTERARBEIT-KERN -> THESIS-CORE, ABWEICHUNGEN -> DEVIATIONS, recherche/ -> research/; comments, help texts and messages in the code translated, logic verified unchanged by an AST comparison; viewer interface and model names in English: basis -> baseline, kubisch -> cubic, synthese2 -> synthetic2, +liquor -> +csf). Identifiers, CLI flags and environment variables keep their German names (glossary in the README), because the commands behind every reported number use them.

**Arm 8 -- result (29 Sep 23:40, evaluation `logs/arm8_t1t2_auswertung_0929.log`):** T1+T2 as extra channels, 60 epochs, CSF rule, cross-selected cell: seed 42 **0.621** (P 0.62, S 0.62, 0.91 FP/case), seed 1 **0.547**, ensemble **0.628** (P 0.67, S 0.59, 0.70 FP/case) versus the baseline ensemble 0.647. Paired baseline ensemble minus zt1t2-s42 +0.026 [-0.031; +0.092]; ensemble minus s42 +0.007 [-0.025; +0.041]. The two extra channels push the cross-selected threshold up (0.6-0.8 instead of 0.3) and trade sensitivity for precision without an F1 gain. **Pre-registered rule: upper bound of the single-sequence solution -- the upper bound is not above the single-sequence chain.** T1/T2 remain a limitation, not a lever; catalina has no T1 anyway. 27 of the 57 T1 volumes carry NaN rims (nan_to_num), a possible reason why the channel contributes nothing; not pursued further.

**Arm 6 -- result (29 Sep 23:45, `logs/arm6_mbnet_score_{valdo,t2star}_0929.log`, grid metric `cmb.transfer.evaluate` on `mbnet/<cohort>/als-lauf`):** the original fine-tuned on our folds: VALDO native **0.040** (S 0.40, 45 FP/case), on our grid **0.068** (P 0.04, S 0.36, 22.6 FP/case, median per case 0.00); T2* native and grid **0.277** (P 0.21, S 0.41, 17 FP/case) versus our U-Net 0.663 / chain 0.671. The pre-registered expectation was 0.30-0.45 (VALDO) and 0.40-0.55 (T2*): both clearly missed. **Finding on the cause:** the original's second stage (CDisc student) learned NOTHING in every fold inspected -- validation accuracy 0.50-0.69 on VALDO, 0.49-0.52 on T2* (chance level), validation loss 32-35 at a training loss of 0.43 (folds 0/2 per cohort, `modell/validation_acc_cdisc_student.npz`); the CDisc predictions have exactly the same components as the CDet candidates (56/76/54 components per case), only the original's size filter halves them. The CDet stage itself learns (loss 0.96 -> 0.48, validation Dice 0.9995). So the number is the yardstick "original code, original fine-tuning, our data" -- not the upper bound of what the architecture could do. Do not write "the original reaches 0.07" into the core balance, but "the original's fine-tuning collapses in its second stage on our data". Open: (a) the original without fine-tuning (zero-shot with the delivered weights) as a second reference; (b) CDisc fine-tuning with a frozen trunk (smaller `ftlayers`) or more epochs; (c) check whether the arena SWI run (0.406) had the same chance-level CDisc accuracy. No private case identifier in this note.
Addendum (c), 29 Sep 23:55: the arena SWI run (`mb-arena/ergebnisse/swi/microbleednet-finetune-alle`) reached a CDisc validation accuracy of 0.44 -> 0.63-0.73 (folds 0/1/4, 26-60 epochs) -- weak, but above chance; on VALDO and T2* the CDisc fine-tuning early-stopped after 13-18 epochs at chance level. So the original's second stage is the weak link on all three cohorts, least so on SWI.

**Arm 9, addendum attention (user decision 30 Sep 08:20, pre-registered):** question: do the attention gates hurt by themselves, or was the collapse of a01-attunet (0.463 vs 0.585, 22 Sep) due to its isotropic first level? Run `a11-anisoatt` = a03-aniso + four additive gates (Oktay) on the skips, otherwise identical (60 epochs, gitter_d, seed 42), queued behind the tournament (entry [268]). Evaluation as in tournament 2: fixed cell raw + CSF rule, paired against baseline s42. **Expectation:** within seed noise below the baseline (0 to -0.05), no collapse like a01; cohort 3 not conspicuous. Rule: >= +0.04 -> second seed as in the tournament; otherwise the finding stays "gates add nothing, the geometry was the cause" or, if < -0.08, "gates hurt by themselves".
