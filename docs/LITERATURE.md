# Literature for the CMB pipeline -- what we used, reproduced, or checked (as of 2026-09-21)

Purpose: reading list for the user and basis for the bibliography. Each line states WHAT the source stands for in our work
(section in `WORKLOG.md`) and how reliable the citation is:

- **[Q]** checked at the source -- page retrieved ourselves (text partly extracted by a small reading model) or code read ourselves
- **[R]** from the research notes (`research/*.md`): usually only a search snippet or abstract -- read before citing
- **[G]** standard source from memory -- check authors, year, volume, and pages before citing

No figure from [R] or [G] sources has been verified by us, unless noted otherwise.

**Cross-check 2026-09-23** (`research/literature-check-2026-09-23.md`): the 20 most important entries with status [R] or [G] --
all from tier A or B, plus the evidence for methods actually reproduced -- were checked against the real sources.
**All 20 confirmed**, no correction, no source that could not be found. Two results from this: the open conference question about
Kofler's "blob loss" is resolved (IPMI 2023, Springer LNCS), and eight entries now have volume and page numbers that
were missing here before (none of it was wrong). The checked entries are marked **[Q 23 Sep]** below. About 27 entries remain
unchecked -- above all the nine tournament architectures of tier C and pure idea sources; they are listed in the check file as
candidates for a second round.

## 0 Not all equally important -- reading order

**Tier A -- read first (carry the argument of the thesis; a reviewer will ask about these):**
1. Sudre et al., "Where is VALDO?" -- dataset, official metric, what the challenge teams achieved. Without this none of our numbers can be put into context.
2. Sundaresan et al. 2023 (MicrobleedNet) -- the method we scrutinize; above all the methods and the limitations named by the authors themselves.
3. He et al. 2026 (CenSynCMB) -- current state of the art on the SAME data (74.3), we reproduced three of its ideas; shows where our gap lies (precision, three sequences).
4. Tsuchida et al. 2024 (SHIVA-CMB) -- how a model is trained across many cohorts and both sequences; our external benchmark.
5. Isensee et al. 2021 (nnU-Net) -- the segmentation standard; rationale for anisotropic kernels, normalization, recipe. Expected as a comparison.
6. Greenberg et al. 2009 and Wardlaw et al. 2013 (STRIVE) -- what a microbleed is and what it is confused with; basis for the discussion of false positives.
7. Varoquaux 2018 -- why cross-validation with ~60 cases has large error bars; supports our caution with differences below 0.03.

**Tier B -- for the depth of individual chapters:**
8. Kofler et al. (blob loss) -- the lesion-wise loss (5j). 9. Dou et al. 2016 -- classic two-stage CMB cascade, hard negatives (5d, 5l).
10. "A geometric-to-neural cascade" 2026 and 11. Al-masni et al. 2020 / Liu et al. 2019 -- further two-stage CMB systems for comparison with our second stage.
12. Setio et al. 2016 and 13. Ghafoorian et al. 2017 -- false-positive reduction on centered candidates, location as input (our tissue feature).
14. Billot et al. (SynthSeg, robust variant) -- what the CSF rule and tissue feature are based on, and where the map goes wrong on T2* / SWI.
15. Loy & Zelinsky 2003 (FRST) -- the second input channel. 16. Lakshminarayanan et al. 2017 (Deep Ensembles) -- why averaging over seeds is more stable.
17. Gregoire et al. 2009 (MARS) and Haller et al. 2018 -- location classification and clinical significance. 18. Momeni et al. 2021 -- synthetic bleeds (for the outlook).

**Tier C -- only look up or cite as a reference for a single building block:** U-Net, 3D U-Net, Attention U-Net, UNet++, SegResNet, Swin UNETR, ResNet, Squeeze-Excitation (tournament architectures); InstanceNorm, AdamW, cosine schedule,
V-Net / Dice, Focal Tversky, Focal Loss, OHEM, CenterNet, uncertainty weighting (losses and training); HD-BET, FSL FAST, Frangi, FreeSurfer (tools); Random Forest, Gradient Boosting, Models Genesis, Med3D,
nnDetection, Retina U-Net, test-time augmentation (second stage and inference); Bootstrap, domain-adaptation survey, Bouthillier et al. (statistics); plus all research hits marked [R] that we have not read.

## 1 Data, challenge, clinical definitions

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| Sudre CH et al., "Where is VALDO? VAscular Lesions Detection and segmentatiOn challenge at MICCAI 2021", Medical Image Analysis 2024 | https://arxiv.org/abs/2208.07167 ; evaluation code https://github.com/WhereIsValdo/valdo-eval-2021 | Task 2 dataset, official metric (`code/metric_valdo.py`), rank comparison (2a, 5) | Q |
| VALDO data, licence CC BY-NC-SA 4.0 | Zenodo 10.5281/zenodo.4520773 ; `dev/licence-valdo.md` | terms of use, mandatory attribution | Q |
| Momeni S et al., "Synthetic microbleeds generation for classifier training without ground truth", Comput Methods Programs Biomed 2021; dataset AIBL-SWI with synthetic CMB | https://www.sciencedirect.com/science/article/abs/pii/S0169260721002029 ; data DOI 10.25919/AEGY-NY12 | free SWI dataset, idea of "synthetic bleeds" (`research/datasets.md`) | R |
| Dou Q et al., "Automatic Detection of Cerebral Microbleeds From MR Images via 3D Convolutional Neural Networks", IEEE TMI 35(5), 2016 | https://pubmed.ncbi.nlm.nih.gov/26886975/ ; data http://www.cse.cuhk.edu.hk/~qdou/cmb-3dcnn/cmb-3dcnn.html | classic two-stage cascade with hard negatives (5d, 5l); free SWI dataset | R |
| Greenberg SM et al., "Cerebral microbleeds: a guide to detection and interpretation", Lancet Neurol 8:165-174, 2009 | -- | definition and mimics of CMB | G |
| Wardlaw JM et al., STRIVE, Lancet Neurol 12:822-838, 2013; Duering M et al., STRIVE-2, Lancet Neurol 2023 | -- | size definition 2-5 (up to 10) mm (5i) | G |
| Gregoire SM et al., "The Microbleed Anatomical Rating Scale (MARS)", Neurology 73:1759-1766, 2009 | -- | location classification lobar / deep / infratentorial (`code/gewebekarte.py`, 5i) | G |
| Haller S et al., "Cerebral Microbleeds: Imaging and Clinical Significance", Radiology 287:11-28, 2018 | -- | overview, mimics (calcification, vessels, cavernomas) | G |

## 2 The original under the microscope: MicrobleedNet

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| Sundaresan V et al., "Automated detection of cerebral microbleeds on MR images using knowledge distillation framework", Front. Neuroinform. 17:1204186, 2023 | https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1204186/full | reference method; sampling / loss (10x CMB weighting), limitations stated by the authors (2a, 4.x, `DEVIATIONS-microbleednet.md`) | Q |
| MicrobleedNet code, commit 958f1cb | https://github.com/v-sundaresan/microbleed-detection | comparison of paper against code (FRST radii, ratio='1:1', inpainting); our fork `custom_microbleednet` | Q |
| Sundaresan V et al., Eur Radiol Exp 2025 (size and location of CMB according to MARS) | https://link.springer.com/article/10.1186/s41747-024-00544-z | MARS atlas in MNI space as a possible location source | R |
| Loy G, Zelinsky A, "Fast radial symmetry for detecting points of interest", IEEE TPAMI 25(8):959-973, 2003 | -- | FRST channel (`code/frst.py`; 4.11: without FRST -0.053) | G |
| Frangi AF et al., "Multiscale vessel enhancement filtering", MICCAI 1998 | -- | vessel filter of the inpainting (4.1: removed, +0.136) | G |
| Hinton G, Vinyals O, Dean J, "Distilling the Knowledge in a Neural Network", 2015 | https://arxiv.org/abs/1503.02531 | distillation in the discriminator of the original (not reproduced) | G |

## 3 Recent works on CMB (benchmarks and idea sources)

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| He L, Zhang H, Li K, Saccoh AF, Ingala S, Rehwald R, de Bruijne M, Barkhof F, Davies R, Sudre CH, "CenSynCMB: Centre Maps and Physics-Guided Synthesis for Microbleed Detection", arXiv 2026-07-06 | https://arxiv.org/abs/2607.05325 | new benchmark on VALDO Task 2 (74.3); centre map (5k), reweighting by missed lesions (5l), synthesis; their evaluation protocol in `cmb/analysis/subject_level_f1.py` | Q |
| Tsuchida A et al., SHIVA-CMB, Scientific Reports 2024 | https://www.nature.com/articles/s41598-024-81870-5 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11680838/ ; code https://github.com/pboutinaud/SHIVA_CMB | external benchmark (5a); trained on 60 of the 72 VALDO cases -> fair only on catalina | Q |
| "A geometric-to-neural cascade for cerebral microbleed detection in susceptibility-weighted MRI", 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC13483767/ | two lightweight classifiers in series, removes 76.8% of false positives (motivation for the second stage) | Q (reading model) |
| Liu S et al., "Cerebral microbleed detection using Susceptibility Weighted Imaging and deep learning", NeuroImage 198:271-282, 2019 | https://pubmed.ncbi.nlm.nih.gov/31121296/ | 3D FRST candidates + ResNet with phase image (we have no phase) | R |
| Al-masni MA et al., two-stage YOLO + 3D-CNN, 2020 | https://pubmed.ncbi.nlm.nih.gov/33018167/ ; https://www.sciencedirect.com/science/article/pii/S2213158220303016 ; code https://github.com/Yonsei-MILab/Cerebral-Microbleeds-Detection | two-stage comparison work | R |
| Rashid T et al., "DEEPMIR", Scientific Reports 2021 | https://arxiv.org/abs/2010.00148 | CMB vs. iron deposition (needs QSM) | R |
| Myung MJ et al., J Stroke Cerebrovasc Dis 2021 | https://www.sciencedirect.com/science/article/abs/pii/S1052305721002895 | CSF / white-matter mask as a location feature (cf. our CSF rule 4.3) | R |
| VALDO teams: Kuijf H, MixMicrobleedNet (nnU-Net); MixMicrobleed (Mask-RCNN + U-Net); Chen Z ("Zihao", multi-stage) | https://arxiv.org/abs/2108.01389 ; https://arxiv.org/abs/2108.02482 ; https://github.com/ZihaoChen0319/CMB-Segmentation | what the top of the challenge did differently (`research/valdo-challenge.md`) | R |
| "Toward Automated Detection of Microbleeds with Anatomical Scale Localization" | https://arxiv.org/abs/2306.13020 | location knowledge against false positives (5i: for us, no gain beyond the CSF rule) | R |

## 4 Network architectures of the tournament (4.4)

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| Ronneberger O et al., U-Net, 2015; Cicek O et al., 3D U-Net, 2016 | https://arxiv.org/abs/1505.04597 ; https://arxiv.org/abs/1606.06650 | basic architecture of all networks | G |
| Isensee F et al., "nnU-Net", Nature Methods 18:203-211, 2021 | -- | a02-nnunet; anisotropic kernels / pooling as the model for the winner a03-aniso; deep supervision | G |
| Oktay O et al., "Attention U-Net", 2018 | https://arxiv.org/abs/1804.03999 | a01-attunet, a11-anisoatt | G |
| Zhou Z et al., "UNet++", 2018 | https://arxiv.org/abs/1807.10165 | a06-unetpp | G |
| Myronenko A, "3D MRI brain tumor segmentation using autoencoder regularization" (SegResNet), 2018 | https://arxiv.org/abs/1810.11654 | a09-segres | G |
| Hatamizadeh A et al., "Swin UNETR", 2022 | https://arxiv.org/abs/2201.01266 | a08-swin (collapsed, aborted) | G |
| He K et al., ResNet, 2015; Hu J et al., Squeeze-and-Excitation, 2017 | https://arxiv.org/abs/1512.03385 ; https://arxiv.org/abs/1709.01507 | a05-resse | G |
| Ulyanov D et al., Instance Normalization, 2016; Loshchilov I, Hutter F, AdamW 2017 and SGDR (cosine) 2016 | https://arxiv.org/abs/1607.08022 ; https://arxiv.org/abs/1711.05101 ; https://arxiv.org/abs/1608.03983 | training recipe (4.5) | G |

## 5 Preprocessing and tools

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| Billot B et al., "SynthSeg", Medical Image Analysis 2023; robust variant: Billot B et al., PNAS 2023 | https://arxiv.org/abs/2107.09559 | tissue map: CSF rule (4.3), location of errors (5i), tissue feature of the second stage (5j), `--robust` directly on T2* / SWI | G |
| Isensee F et al., "HD-BET", Human Brain Mapping 40:4952-4964, 2019 | -- | brain masks catalina (4.2, 5n: tight compared to the loose VALDO masks) | G |
| Zhang Y, Brady M, Smith S, FSL FAST, IEEE TMI 20:45-57, 2001; Jenkinson M et al., "FSL", NeuroImage 2012 | -- | bias correction `fast -B` (4.9: without -0.022) | G |
| Fischl B, "FreeSurfer", NeuroImage 2012 | -- | runtime environment of SynthSeg (FreeSurfer 8.2.0) | G |

## 6 Losses and training levers (5g, 5j-5l)

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| Kofler F et al., "blob loss: instance imbalance aware loss functions for semantic segmentation", 2022/2023 (IPMI 2023 (Information Processing in Medical Imaging), Springer LNCS -- conference confirmed at the source on 2026-09-23) | https://arxiv.org/abs/2205.08209 | `--verlust blob` (5j): small CMB 0.52 -> 0.64, F1 not better | R |
| Milletari F et al., "V-Net" (Dice loss), 2016 | https://arxiv.org/abs/1606.04797 | BCE + Dice in the recipe; cause of the collapse at 1:2 (5g) | G |
| Abraham N, Khan NM, "A Novel Focal Tversky loss function ...", ISBI 2019 | https://arxiv.org/abs/1810.07842 | `cv5.loss_focal_tversky` (tested earlier) | R |
| Zhou X et al., "Objects as Points" (CenterNet), 2019; Kendall A et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018 | https://arxiv.org/abs/1904.07850 ; https://arxiv.org/abs/1705.07115 | background for the centre map and its weighting in CenSynCMB (5k) | G |
| Shrivastava A et al., OHEM, 2016; Lin T-Y et al., Focal Loss, 2017 | https://arxiv.org/abs/1604.03540 ; https://arxiv.org/abs/1708.02002 | hard negatives (5l); Focal Loss tested and rejected as unsuitable (candidates are balanced) | G |
| Further instance losses from the research: "Learning to Look Closer" (CC-DiceCE), ICI loss, CATMIL | https://arxiv.org/abs/2511.17146 ; https://arxiv.org/abs/2304.06229 ; https://arxiv.org/abs/2604.08015 | not tested | R |

## 7 Second stage, ensemble, test-time augmentation (5d, 5j)

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| Setio AAA et al., "Pulmonary Nodule Detection in CT Images: False Positive Reduction Using Multi-View Convolutional Networks", IEEE TMI 35:1160-1169, 2016 | https://pubmed.ncbi.nlm.nih.gov/26955024/ | model for false-positive reduction on centered candidates | R |
| Ghafoorian M et al., "Deep multi-scale location-aware 3D CNNs for automated detection of lacunes", NeuroImage: Clinical 2017 | https://arxiv.org/abs/1610.07442 | location + multiple scales as input to the second stage (our tissue feature) | R |
| Breiman L, "Random Forests", Machine Learning 2001; Friedman JH, Gradient Boosting, Annals of Statistics 2001 | -- | classifiers in `cmb/stage2/` | G |
| Zhou Z et al., "Models Genesis", 2019/2021; Chen S et al., "Med3D", 2019 | https://arxiv.org/abs/2004.07882 ; https://arxiv.org/abs/1904.00625 | pretrained 3D encoders (not tested) | R / G |
| Baumgartner M et al., "nnDetection", MICCAI 2021; Jaeger PF et al., "Retina U-Net", 2018 | https://arxiv.org/abs/2106.00817 ; https://arxiv.org/abs/1811.08661 | detection head as an alternative (deferred) | G / R |
| Lakshminarayanan B et al., "Deep Ensembles", NeurIPS 2017 | https://arxiv.org/abs/1612.01474 | seed ensemble (`cmb/analysis/ensemble.py`) | G |
| Wang G et al., "Aleatoric uncertainty estimation with test-time augmentation ...", Neurocomputing 2019 | https://www.sciencedirect.com/science/article/pii/S0925231219301961 | mirror TTA (`cmb/analysis/tta_predict.py`) | R |
| Further from the research (not read): two-stage system for brain metastases (Eur Radiol 2023), GBT for lung nodules, uncertainty against false positives (liver), anatomical post-processing for aneurysms | https://pubmed.ncbi.nlm.nih.gov/36695903/ ; https://arxiv.org/abs/1708.05897 ; https://arxiv.org/abs/2206.10911 ; https://arxiv.org/abs/2507.00832 | idea sources, see `research/second-stage-fp-reduction.md` | R |

## 8 Transfer between cohorts, statistics

| Source | Link | What it's for in our work | Status |
|---|---|---|---|
| "Studying Robustness of Semantic Segmentation under Domain Shift in Cardiac MRI" | https://arxiv.org/abs/2011.07592 | mixing of domains (5e, 5h, 5n) | R |
| Guan H, Liu M, "Domain Adaptation for Medical Image Analysis: A Survey", IEEE TBME 2022 | -- | overview (recommended reading, not used by us) | G |
| Efron B, Tibshirani R, "An Introduction to the Bootstrap", 1993 | -- | paired bootstrap across cases (section 3) | G |
| Varoquaux G, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage 180:68-77, 2018; Bouthillier X et al., "Accounting for Variance in Machine Learning Benchmarks", 2021 | -- ; https://arxiv.org/abs/2103.03098 | context for our seed and fold noise (recommended reading) | G |

Complete research tables with all (including unchecked) hits: `research/valdo-challenge.md`, `research/sundaresan-vs-valdo.md`,
`research/datasets.md`, `research/segmentation-methods.md`, `research/second-stage-fp-reduction.md`.
