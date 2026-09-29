# Vessel cross-sections as false positives: what the literature yields for our case (2026-09-29; sonnet subagent, 10 web searches; check numbers against the source before citing)

Constraints: magnitude only (T2*/SWI), no phase, 2-5 mm slices resampled to 0.5 x 0.5 x 1 mm. Already measured and
not helpful: run length along z (Momeni only), Frangi (reverses), hard negatives online (±0), naive artificial
mimics (Gaussian tubes: both models 0.00), inpainting (erases haemorrhages).

| # | Method | Source | Data | Number | Transferable? |
|---|---|---|---|---|---|
| 1 | **Physics-guided synthesis** of vessel/calcification mimics and haemorrhages (dipole-kernel rendering, randomised field strength 1.5-3 T, TE 20-40 ms) + centre map | CenSynCMB, arXiv 2607.05325 (2026) | VALDO T2* (72), AIBL SWI (370, 1.75 mm) | Ablation VALDO: F1 71.7 -> 74.3, precision 74.6 -> 79.3, **FP per case 1.4 -> 0.8** through the synthesis | YES, same dataset. Difference from our arm 1: their mimics are PHYSICALLY rendered (dipole pattern, blooming), ours were smooth tubes that neither model confused |
| 2 | Geometric pre-filtering (aspect ratio = vessel) before two 3D ResNets | "geometric-to-neural cascade", PMC13483767 | 30 SWI, 11 scanners | FP 1157 -> 269 (-77%), sensitivity 93 -> 71%, F1 0.69 | NO: aspect ratio is exactly the feature that flips on our grid |
| 3 | Mimic search with YOLO, hard FP as an extra class for the 3D CNN | PMC12410576 | ADNI GRE (1517), AIBL SWI, MAS/OATS SWI | AIBL FP per scan ~9 -> 1.2; but on ADNI F1 0.930 -> 0.902 | partly; corresponds to our hard negatives (±0) -- cohort-dependent |
| 4 | Multi-task teacher with **anatomy conditioning** | Front. Neuroinform. 2023, doi 10.3389/fninf.2023.1204186 | incl. **OXVASC 2D-T2* 5 mm (74)** | OXVASC: TPR 90%, **0.9 FP per case, precision 84%** | MOST RELEVANT reference value for thick slices; read which anatomy conditioning (beyond our CSF rule) |
| 5 | DEEPMIR: T2w + SWI + QSM jointly | Rashid 2021, Sci Rep 11:14124 | MESA, 24 | sensitivity ~0.8, precision ~0.6 | no (QSM mandatory) |
| 6 | 2.5D FRST candidates + V-Net, SWI + high-pass phase | Liu 2019, NeuroImage: Clinical | SWI + phase | sensitivity 95.8%, precision 70.9%, 1.6 FP per case | only with phase -- the upper comparison value |
| 7 | Learned vessel segmentation as an extra channel (VesselBoost or similar) | -- | -- | **no evidence with a number found**; SHIVA-CMB mentions vesselness/CSF masks only qualitatively | open gap, untested |
| 8 | Super-resolution along z | arXiv 2608.06311 (WMH), PubMed 42155350 (SWI) | -- | no CMB number; warning: SR can erase small lesions | no |
| 9 | Magnitude versus QSM | ISMRM 2020 #3223 (uncertain) | -- | SWI magnitude finds 2.5x as many CMB as QSM; only 15% of CMB < 5 mm3 visible on QSM | phase is not automatically helpful for SMALL haemorrhages |

## Assessment against our measurements
- The only proven, reproducible lever without phase is No. 1 -- and specifically in the version we did NOT build: mimics with physical rendering (susceptibility source -> dipole field -> signal loss at TE, with blooming). Our Gaussian tubes were rated 0.00 by both models, i.e. never confused; the dipole-rendered ones from CenSynCMB cut the false positives by nearly half. Effort: forward model (dipole-kernel convolution, TE/B0 randomisation, conversion onto our grid with slice thickness), 2-3 days; then the same measurement protocol as 5ae-1b. Risk: their gain came together with T1+T2+T2* and the centre map; for us the centre map did not help.
- No. 4 is the only value on 5 mm T2* -- precision 0.84 at 0.9 FP per case. If this was achieved with "anatomy conditioning" without phase, it is our benchmark for thick slices (our T2*: precision 0.70, 3.0 FP per case). Read the paper.
- No. 7 (learned vessel mask from magnitude as a channel) is unproven but the obvious idea with "other information"; needs a vessel segmentation network for SWI/T2* (pretrained or on few cases), medium effort, outcome open.
- No. 9 puts the phase wish into perspective: for small haemorrhages, magnitude is more sensitive anyway; phase helps precision.

## Recommendation (order)
1. Read papers No. 4 and No. 1 in full text (anatomy conditioning; rendering details of the mimics). No GPU.
2. If No. 1 is reproducible: mimic synthesis with dipole rendering as arm 1c, rule as in 5ae-1b, two seeds.
3. No. 7 as arm 7 only if a usable vessel segmentation network is found.
Not: aspect-ratio filter, super-resolution.

## Addendum 29 Sep midday: the two full texts (subagent; WebFetch extraction, check citations against the PDF before adopting)

**CenSynCMB (arXiv 2607.05325v1, He, ..., Sudre):** Rendering = dipole kernel D(k) = 1/3 - kz^2/|k|^2, static dephasing
phi = gamma * dBz * TE, magnitude = magnitude of the voxel-averaged complex signal; B0 1.5-3 T and TE 20-40 ms randomised; vessel
mimics = "elongated anisotropic masks", calcification = "compact low-signal sources"; mimics only as label 0, no own class;
positives from the priors of the training fold (number per subject, centres, diameter, local intensity change). NOT specified:
susceptibility in ppm, number per image, additive/replacing, anatomy prior, partial volume of the slice thickness (everything
resampled to 1 mm isotropic), code. **Ablation Table IV (VALDO, n = 72, T1+T2+T2*): real 71.7 / +CMB 73.1 / +CMB+mimics 74.3 F1; precision
74.6 -> 79.3; FP per case 1.4 -> 0.8 -- "all p > 0.05".** Consequence: the dipole reproduction (arm 1c) moves down the list -- 2-3 days for an
effect that is not even secured in the source itself, and with two more sequences than we have.

**Sundaresan et al. 2023, Front. Neuroinform. (knowledge distillation; code github.com/v-sundaresan/microbleed-detection = our
original!):** candidate U-Net (image + FRST) -> teacher (features + segmenter + classifier) -> student via distillation.
Preprocessing with vessel inpainting (for us -0.136). **Post-processing (2.4): volume < 2.5 mm3 removed; ellipticity > 0.2 removed
(tubes); candidates < 5 mm from the brain mask edge removed (sulci).** OXVASC (74 cases, 366 CMB, 0.9 x 0.8 x 5 mm, magnitude
ONLY): cluster TPR 0.93 -> 0.91 -> 0.90, FP per case 195.7 -> 10.1 -> **0.9**, precision 0.02 -> 0.29 -> **0.84** across the three stages.
UKBB ablation: post-processing FP 14.7 -> ~0.5 at TPR 0.90 -> 0.83. Consequence: the **edge rule (>= 5 mm from the brain edge)**
is unmeasured for us and cheap -- measurement on 29 Sep midday (`border_*.json`); ellipticity flips on our grid (Frangi finding).
Caution when comparing the TPR (cluster rule, different cohort, median 3 CMB per case).
