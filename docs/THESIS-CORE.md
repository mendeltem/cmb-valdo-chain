# Core of the master's thesis: from MicrobleedNet to our chain (draft 2026-09-29, figures as of: WORKLOG.md 5ae)

Guiding idea: We re-implemented MicrobleedNet (Sundaresan et al. 2023) on VALDO and then swapped out EVERY component individually against
a paired measurement. Only what was confirmed or reproducibly better stayed. The thesis tells this chain,
not the 40 attempts that brought nothing (those are listed as a table in the appendix).

Metric throughout: lesion F1 (pooled, 26-connectivity neighbourhood, touch rule), 5-fold cross-validation on 57 VALDO cases
(139 lesions), paired bootstrap over cases; "confirmed" = 95% interval excluding zero. Noise level of a single
seed ~0.08, hence two seeds per decision.

## 0 Abstract (draft 2026-09-29; check figures against WORKLOG.md before submission)

**German.** Cerebral microbleeds (CMB) are millimetre-sized, hypointense lesions on T2*-weighted and susceptibility-weighted
MRI, whose manual counting is laborious and poorly reproducible. The starting point of this thesis is MicrobleedNet
(Sundaresan et al. 2023), a two-stage, publicly available method. We re-implemented it on the public VALDO dataset
(72 cases, three cohorts with 0.8 to 4 mm slice thickness) and then swapped each component of the chain individually against a
paired, case-bootstrapped measurement in 5-fold cross-validation; only changes that were confirmed or reproduced better with a
second seed stayed. Seven changes contributed: dropping the original's vessel inpainting
(+0.14), an anisotropic 3D U-Net instead of an isotropic attention U-Net (+0.16), a shared training grid of 0.5 x 0.5 x 1 mm,
the retained FRST channel (+0.05), a fixed sampling scheme with a bounded budget per case, an anatomical post-processing rule via
SynthSeg tissue maps (CSF rule, +0.06 with no true lesion lost) and a cross-cohort-trained 3D-CNN candidate classifier
as second stage. Lesion F1 rose from 0.245 to 0.647; on two in-house clinical cohorts (T2*, 75 cases; SWI, 61 cases)
the chain with a cohort-specific U-Net and a shared classifier reached 0.67, against 0.41 for the original fine-tuned on the
same folds. On an external public SWI cohort (57 cases) the chain trained purely on VALDO scored 0.61 to 0.65 without any
adaptation (median per subject 0.67). About forty further interventions -- loss functions, augmentation, intensity harmonisation, synthetic
lesions, longer training, second-stage variants, shape filters against vessels -- raised sensitivity, not F1: the
remaining bottleneck is separating lesion from vessel cross-section with a single magnitude sequence, especially under heavy
loads of small, low-contrast lesions. The resulting chain is fully trainable from public data and is released with code,
commands and all negative results.

**English.** Cerebral microbleeds (CMB) are millimetre-sized hypointense lesions on T2*-weighted and susceptibility-weighted MRI whose
manual counting is laborious and poorly reproducible. Starting from MicrobleedNet (Sundaresan et al. 2023), a public two-stage
method, we re-implemented the pipeline on the public VALDO dataset (72 cases, three cohorts, 0.8-4 mm slices) and then replaced
each component in turn, accepting a change only if it improved the paired, case-bootstrapped lesion-level F1 in 5-fold cross-validation
and survived a second random seed. Seven changes held: dropping the original's vessel inpainting (+0.14), an anisotropic 3D U-Net
instead of an isotropic attention U-Net (+0.16), a common 0.5 x 0.5 x 1 mm training grid, the retained radial-symmetry channel (+0.05),
a fixed sampling scheme with a bounded budget per case, an anatomical post-processing rule from SynthSeg tissue maps (CSF rule, +0.06
with no true lesion lost) and a cross-cohort 3D-CNN candidate classifier as second stage. Lesion F1 rose from 0.245 to 0.647; on two
in-house clinical cohorts (T2*, 75 cases; SWI, 61 cases) the chain with cohort-specific U-Nets and a shared classifier reached 0.67
against 0.41 for the original fine-tuned on the same folds. On an external public SWI cohort (57 cases) the VALDO-only chain scored
0.61-0.65 without any adaptation (median per subject 0.67). About forty further interventions -- loss functions, augmentation,
intensity harmonisation, synthetic lesions, longer training, second-stage variants, shape filters against vessels -- raised sensitivity
but not F1: the remaining bottleneck is separating microbleeds from vessel cross-sections with a single magnitude sequence, most of all
under heavy loads of small, low-contrast lesions. The resulting chain is trainable from public data alone and is released with code,
commands and all negative results.

## 1 Starting point: MicrobleedNet on VALDO
Two-stage (CDet 48 mm cube, CDisc 24 mm cube, teacher-student), vessel inpainting as preprocessing, FRST as
additional channel, inverted scale 1 - I/max. Re-implementation with Attention U-Net and the original recipe on VALDO-T2*: **F1 0.245**
(precision 0.20). Three errors in the original code documented (DEVIATIONS-microbleednet.md A1-A3).

## 2 The seven changes that stayed (each with the measurement that supports it)
| No. | Component | Original | Our chain | Measurement | Why |
|---|---|---|---|---|---|
| 1 | Vessel inpainting | yes (before training) | **none** | **+0.136 confirmed** (4.1) | Inpainting hit 109 of 236 lesions, 17 became invisible; the network detects vessels on its own |
| 2 | Architecture | Attention-U-Net-like (CDet) | **anisotropic 3D U-Net a03-aniso** (first stage, in-plane only) | **+0.159 confirmed** in the tournament over 9 networks; confirmed on clean data (nnU-Net 0.458, Attention 0.463, a03 0.585) | thick slices (0.8-5 mm) require anisotropic convolution |
| 3 | Grid | original resolution | **0.5 x 0.5 x 1 mm**, brain crop | z = 1 vs 2 mm +0.036 (4.8) | fine z-sampling separates "ends" from "continues" |
| 4 | FRST channel | yes | **kept** | +0.053 (4.11) | only preprocessing lever with a confirmed gain |
| 5 | Sampling | 1:x class ratio | **60% lesion / 25% brain / 15% arbitrary**, 800 per epoch, 60 epochs, checkpoint selected on inner fold | 1:2 collapses (5g); 120 epochs -0.022 confirmed (5h) | budget per case is the bottleneck, not epochs |
| 6 | Post-processing | threshold | **fixed cell 0.3 / 2 mm3 + CSF rule** (SynthSeg: majority in CSF -> discard) | **+0.06, 0 lesions lost** (4.3) | 98 of 176 false positives lay in free CSF |
| 7 | Second stage | CDisc per dataset, teacher-student | **one shared 3D-CNN candidate classifier across all cohorts** (16 mm cube, image + FRST + probability), own U-Net per cohort | SWI **+0.045 confirmed / +0.058 (10 seeds)**; VALDO / T2* equal to the CSF rule; teacher-student +0.003 (5z) | the pool of all cohorts is large enough, per-cohort it was not; on SWI it replaces the hard rule, which deletes 74 true lesions there |

Result of the chain on VALDO: **0.245 -> 0.647** (two-seed ensemble with CSF rule; mean over single seeds 0.605 ± 0.034).

## 3 Transfer to the in-house clinical cohort (the actual purpose)
| Strategy | catalina-T2* (75 cases, 828 CMB) | catalina-SWI (61, 652) |
|---|---|---|
| VALDO model without adaptation | 0.622 | 0.305 |
| mixed / pretrained + fine-tuned | -0.02 to -0.045 confirmed | -0.045 confirmed |
| **own training per cohort, 60 epochs** | **0.663** | **0.626** |
| + shared classifier | **0.671** | **0.672** |
Finding: U-Nets per cohort, classifier shared. And the classifier, trained ONLY on VALDO, is without fine-tuning
just as good (SWI 0.671, T2* 0.648) -> a purely public, publishable chain.

## 4 Honest check outside cross-validation
| Check | Result |
|---|---|
| 15 held-out VALDO cases, one-time, frozen chain | pooled 0.468; one case carries 73 of the 97 lesions (sensitivity there 0.20); without it 0.731; median per case 0.667 |
| Momeni/CSIRO, external public SWI cohort (57 cases, 146 lesions), without adaptation | VALDO U-Net + CSF rule 0.613, + run-length rule 0.645; SWI U-Net + VALDO classifier 0.630; median per subject 0.667; sensitivity ~0.8 without rules |
Limit of the chain: heavy load of small, low-contrast lesions (sensitivity 0.20 on the hardest case).

## 5 What did not help (a table in the appendix, here only the sentence)
No training lever (blob loss, centre map, fnrw, hard negatives, augmentation, 120 epochs, patch size, synthetic
lesions in two versions), no intensity harmonisation (z, landmarks, p99.5, cubic), no variant of the second stage
(two-scale, teacher-student, pretraining, more candidates) raised the F1. The pattern was always the same: more sensitivity,
more false positives. The bottleneck is separating lesion / vessel with a single sequence -- and shape filters (run length, Frangi)
fail on the anisotropic grid, because blooming turns a lesion into a short cylinder.

## 6 Proposed outline (short)
1 Introduction (CMB, MicrobleedNet as starting point, goal: in-house clinical cohort) · 2 Data (VALDO 3 cohorts, catalina T2*/SWI,
Momeni) and metric · 3 Re-implementation and errors of the original · 4 The seven changes (Table 2, one paragraph each with a figure
from the slice viewer) · 5 Transfer (Table 3) · 6 Check outside the CV (Table 4) · 7 Limits and negative findings
(one paragraph + appendix table) · 8 Conclusion: publishable chain, limit under heavy load.
Figures: (a) inpainting deletes lesions (4.1), (b) chain schematic, (c) example cases from the viewer (reference against
basis+CSF rule), (d) bar chart of the seven changes, (e) Momeni example. Sources for all figures: WORKLOG.md, protokoll.md,
REPLICATION.md (verbatim commands), OVERVIEW-tests-2026-09-29.md (all tests).

## 7 Which figures the thesis needs -- and which it does not (user question 29 Sep)

**Primary measure (one figure per comparison):** lesion F1, pooled over all cases (26-connectivity neighbourhood, touch rule), at the fixed
cell 0,3 / 2 mm3, ALWAYS paired against the baseline with bootstrap over cases (10 000 draws, 95% interval). Without an interval no
figure in the thesis is interpretable; without a second seed nothing below 0,08 is interpretable for the U-Net (section 3 in WORKLOG.md).
**Secondary measures (always report together, never alone):** sensitivity, precision, false positives per case (the pattern "sensitivity up,
F1 down" is the most important finding of the thesis and is visible only through these three), plus median F1 per subject under the
challenge rule (comparability with the VALDO paper and CenSynCMB) and sensitivity per size class (< 7 / 7-14 / > 14 mm3).
**AUC:** only where a CONTINUOUS quantity separates candidates -- candidate features (ROC figure) and the classifier's output.
A voxel-level AUC of the detector is meaningless at 0,01% positive voxels and is not reported.
**Loss:** not an outcome measure. One figure: training loss at 60 / 90 / 120 epochs next to the F1 -- the loss keeps falling, the F1
does not (budget per case, 5h). That is the argument against "train longer".
**Volume (ml):** descriptive (reference against prediction per cohort, one table), not as a target quantity -- CMB volume depends on
blooming, clinically what counts is the NUMBER. Instead: counting error per case (|predicted - reference|) and case accuracy (does the case
have any lesions at all), both are in `evaluate`.
**Spread:** seed noise of the U-Net (three seeds: 0,646 / 0,590 / 0,578) and second stage (10 seeds: sd 0,003-0,009)
quantified once each; after that every decision with two seeds.

**Protocol "one change against the baseline":** yes, exactly that -- and tell it in the thesis as a CUMULATIVE chain (Table 2), not as
40 individual comparisons: 0,245 (re-implementation) -> 0,404 (without inpainting + anisotropic network, tournament) -> 0,487 (+ CSF rule) -> 0,585 raw /
0,646 with rule (grid d, 60 ep., FRST, sampling) -> 0,647 (two seeds). Each step has its paired figure against the
predecessor. The rules that keep the procedure honest: write down the expectation and decision rule BEFORE the run; touch the test set only
once; never reinterpret anything after the fact; report all negative results (appendix). What was NOT done and is
in the outlook: a backward ablation of the final chain (removing each change individually again) -- partly available (without FRST -0,053,
without CSF rule -0,06, Attention instead of aniso -0,12), partly not (putting inpainting back in).
