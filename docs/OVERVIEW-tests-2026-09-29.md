# Overview of all tests of the CMB pipeline (state 2026-09-29, 08:15) and the best chain for catalina

All figures are lesion-wise F1 (pooled, 26-connectivity, touch rule), paired across cases with bootstrap; "confirmed" =
95% interval excludes zero. VALDO = 57 CV cases, catalina-T2* = 75 cases / 828 CMB, catalina-SWI = 61 / 652.
Noise level: single seed of the U-Net ~0.08 (three seeds 0.646 / 0.590 / 0.578), second stage sd 0.003-0.009 (10 seeds).
Sources: WORKLOG.md (section numbers in parentheses), mb-arena/protokoll.md, REPLICATION.md.

## 1 What was measured

### Data chain / preprocessing (VALDO, paired against the respective baseline)
| Lever | Result | Verdict |
|---|---|---|
| Omit the original's vessel inpainting (4.1) | **+0.136 confirmed** | adopted |
| Keep FRST as second channel (4.11) | **+0.053** (0.588 vs 0.535) | adopted |
| Resampling z = 1 mm instead of 2 mm (4.8) | +0.036 | adopted |
| Keep bias correction (FSL FAST -B) (4.9) | +0.022 n.s. | adopted |
| Percentile scale p99.5 instead of maximum (4.9) | +0.025 n.s., one seed | not adopted |
| z-normalisation per case (5ac) | VALDO -0.005, zero-shot T2* -0.004 / SWI -0.022, all n.s. | not adopted |
| Landmark alignment Nyul-Udupa (5ac) | VALDO -0.064, zero-shot **-0.048 / -0.034 confirmed**, mixed VALDO share 0.355 | not adopted |
| Cubic instead of linear interpolation (5ae arm 2) | ensemble -0.024 n.s.; sensitivity small 0.50 -> 0.68, FP 0.82 -> 1.25 | not adopted |
| Tissue map (SynthSeg) as input channels (5m) | -0.058 | not adopted |
| SynthSeg mask instead of VALDO mask (`--maske synthseg`) | needed for cohort 3 (head mask), otherwise equal | tool |

### First stage: architecture and training (VALDO)
| Lever | Result | Verdict |
|---|---|---|
| Tournament 9 networks, 40 ep. (4.4) | a03-aniso 0.404 raw vs Attention U-Net 0.245: **+0.159 confirmed**; swin, resse, segres, flat confirmed worse | a03-aniso |
| nnU-Net / Attention U-Net on clean data, 60 ep. (5s) | 0.458 / 0.463 vs 0.585 raw | a03-aniso stays |
| Epochs 60 vs 120 (5h) | VALDO -0.029 raw, catalina-T2* -0.022 confirmed | 60 ep. |
| last checkpoint instead of selection on inner fold (5o) | mean drops 0.618 -> 0.604 | checkpoint selection stays |
| Patch 24 / recipe / 48 mm (5t) | 0.369 / 0.585 / 0.569 (24 mm -0.216 confirmed) | 38 x 31 x 47 mm |
| ratio positive:negative 1:2 (5g) | collapses (empty prediction) | 60/25/15 stays |
| Augmentation rotation/scale/contrast (4.7, 5p) | thick-slice -0.127, intensity -0.108; zero-shot likewise negative | mirroring only |
| blob loss (5j) | -0.044 n.s.; catalina-T2* -0.029 confirmed; SWI -0.003 (small CMB 0.37 -> 0.44) | no |
| Centre map as auxiliary output (5k) | -0.054 confirmed | no |
| fnrw loss (5l) | -0.030 n.s. | no |
| hard negatives online (5l) | +0.000 | no |
| synthetic bleeds + mimics, 1st version (5ae arm 1) | ensemble **-0.130 confirmed**; mimics rated 0.00 by both models | no |
| synthetic bleeds, budget preserved, 2nd version (5ae-1b) | ensemble **-0.072 confirmed**; sensitivity small 0.50 -> 0.57, FP 0.82 -> 1.21 | no, arm closed |
| Ensemble over seeds (5e, 5u) | ~ best member, +0.02-0.04 above the mean, flattens after 2 | **ship 2 seeds** |
| Mirror TTA (5e) | VALDO +0.02, catalina-T2* -0.002 | no |

### Transfer to catalina (first stage)
| Strategy | catalina-T2* | catalina-SWI | Verdict |
|---|---|---|---|
| VALDO model zero-shot | 0.622 (s1 0.638) | 0.305 (s1 0.311) | reference value |
| own training 60 ep. | **0.663** | **0.626** | **baseline** |
| fine-tuning from the VALDO model, 30 ep. | = own training | rather worse | no |
| mixed VALDO + catalina, equal budget per case (5h) | 0.641 (e120), +0.001 vs own e120 | -- | no |
| pretraining on all + fine-tuning (5e) | -0.039 confirmed | -0.045 confirmed | no |
| mixed with landmarks (5ac) | 0.630 | -- | no |
| second seed own training | -- | s1 0.613 (-0.022 n.s.) | ensemble sensible, not measured |

### Second stage (candidates from threshold 0.15 / 1 mm3, `--keep-csf`)
| Variant | VALDO | catalina-T2* | catalina-SWI | Verdict |
|---|---|---|---|---|
| threshold only (fixed cell 0.3 / 2 mm3) | 0.585 raw | 0.654 | 0.626 | reference |
| + hard CSF rule (4.3) | **0.646** (+0.06, 0 TP lost) | 0.670 | 0.614 (deletes 74 real ones) | VALDO/T2* yes, SWI **never** |
| per cohort: logistic / forest / boosting / CNN (5d) | ≤ threshold | ≤ threshold | ≤ threshold | no |
| shared forest / logistic (5e) | ±0.04 with pool composition | -- | gain dissolved with independent stage 1 | no |
| **shared 3D CNN, 16-mm cube** (5e, 10 seeds 5z) | 0.632 (= rule) | 0.671 (= rule) | **0.672 (+0.058 vs rule, +0.045 confirmed)** | **yes** |
| two-scale 16 + 28 mm (5r) | -0.030 confirmed | +0.016 / +0.019 (confirmed once) | +0.016 / +0.013 n.s. | not adopted (VALDO rule) |
| teacher-student alpha 0.3, 10 seeds (5z) | +0.001 | +0.003 | +0.002 | no |
| self-supervised pretraining of the encoder (5ab) | +0.012 | -0.005 | -0.011 | no |
| union of the candidates of three seeds (5y) | -0.039 | 0.000 | +0.006 | no |
| union of majority candidates only (5y) | -0.015 | 0.000 | +0.007 | no |
| candidates from more sensitive first stages as addition (5s) | chain 0.594 vs 0.625 | -- | -- | no |
| **VALDO classifier zero-shot** (5ae arm 4) | -- | 0.648 (-0.022, threshold too strict) | **0.671 (= pool)** | publishable |
| fine-tuning the VALDO classifier 10 / 25 / 50 / 100% | -- | -0.064 / -0.025 / -0.004 / 0.000 | -0.051 / -0.021 / -0.009 / -0.012 | neutral only from 50% |

### External validation and test set
| Measurement | Result |
|---|---|
| 15 held-out VALDO cases, frozen chain (5ad) | pooled 0.468 (one case with 73 CMB, S 0.20), without it 0.731, median per case 0.667 |
| Momeni/CSIRO (57 SWI, 146 points), zero-shot without rule | VALDO U-Net 0.525, SWI U-Net 0.491, sensitivity ~0.8 |
| ditto with CSF rule | 0.613 / 0.555, median per subject 0.667; the rule costs sensitivity here 0.80 -> 0.64 |
| ditto with VALDO classifier (`apply.py`) | VALDO U-Net 0.607 (= rule), **SWI U-Net 0.630 (+0.106 confirmed vs rule)** |

### Tools that have NOT proven themselves as deciders
CPU pre-check of the intensity alignment (5aa/5ac): ranking disproved by the GPU run. Candidate ceiling (5w/5y): a higher ceiling
brought nothing. Robustness probe (5v): confirmed once. All three describe, none decides.

## 2 The best chain for catalina (highest measured F1)

**Preprocessing (per case, both sequences the same):** HD-BET brain mask; FSL FAST -B bias correction (once); inverted scale
1 - I / max in the brain (no z-normalisation, no landmarks, no p99.5); NO vessel inpainting; resampling to 0.5 x 0.5 x 1.0 mm,
linear, brain crop (`code/gitter_c_bauen.py`, `cmb.transfer.build_private_grids`); FRST on the grid as second channel
(`code/frst.py`, radii 2/3/4/6 voxels); SynthSeg --robust on the native bias-corrected image for the tissue class (only for the second stage).

**First stage (per cohort OWN training, never mixed, never fine-tuned):** a03-aniso (anisotropic 3D U-Net, 22.5 million
parameters, input T2*/SWI + FRST), 60 epochs, patches 38 x 31 x 47 mm, 800 per epoch, sampling 60% lesion / 25% brain /
15% arbitrary, mirroring only, AdamW 3e-4 with cosine schedule, BCE + Dice with deep supervision, checkpoint chosen on the
inner fold. Measured: T2* 0.663, SWI 0.626 (threshold alone). To ship: mean of the five fold models, and if possible two
seeds averaged (not measured on catalina; on VALDO +0.02-0.04 above the mean, and which seed comes out ahead cannot be known in advance).

**Post-processing stage 1:** threshold 0.3, minimum size 2 mm3, giant components removed (fixed cell; cross-wise selection varies by 0.02-0.05).

**Second stage:** candidates at threshold 0.15 / 1 mm3 with tissue class (`cmb.stage2.candidates --keep-csf`); shared
3D CNN on the pool of all cohorts (`cmb.stage2.pooled --models cnn`, 16-mm cube from image + FRST + probability, three
members averaged, threshold per cohort from inner folds). Measured: **T2* 0.671, SWI 0.672**. On T2* the hard
CSF rule is equivalent (0.670) and cheaper; on SWI the hard rule must NOT run (0.614, deletes 74 real bleeds).
If catalina alone counts: the two-scale CNN (16 + 28 mm) is at 0.687-0.691 on T2* (+0.016 / +0.019, confirmed once)
and at 0.678-0.686 on SWI (+0.013 / +0.016 n.s.) -- discarded following the VALDO rule (there -0.030), for a pure
catalina release the highest measured figure, but only robust with further seeds.

**Publishable variant (without private candidates in training):** VALDO classifier zero-shot (`cmb.stage2.apply`):
SWI 0.671 (equal), T2* 0.648 (-0.022, because the VALDO threshold 0.50 is too strict for T2* -- a locally chosen threshold
would presumably close this, but it is not measured).

**Do not do:** mixing or fine-tuning the U-Net between cohorts; hard CSF rule on SWI; z-normalisation / landmarks;
augmentation other than mirroring; 120 epochs; blob / centre / fnrw; TTA on catalina; synthetic bleeds.
