# Did MicrobleedNet learn from VALDO? And what do we learn from both? (Claude, 2026-09-19, frugal research)

Sources: Sundaresan et al., Front. Neuroinform. 2023 (https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1204186/full),
evaluated by a small reading model, quotes taken verbatim, not checked line by line by me;
VALDO: research/valdo-challenge.md; code: commit 958f1cb, read myself. Sundaresan et al. 2025 (Eur. Radiol. Exp.,
https://link.springer.com/article/10.1186/s41747-024-00544-z) only from the abstract.

## Read
- Timing: MicrobleedNet preprint medRxiv November 2021, VALDO challenge MICCAI 2021 (paper 2022/2024) -- arose in parallel.
- In the 2023 paper, VALDO is used ONLY as: "20 subjects for hyper-parameter tuning". No training, no test on VALDO,
  no comparison with the VALDO teams, none of their methods cited. Data: UK Biobank (78, SWI/QSM, 0.8x0.8x3 mm),
  OXVASC (74, T2*-GRE, 0.9x0.8x5 mm), TICH2 (115, SWI), SHK (20, SWI).
- Results there (different data, different metric than ours): cluster TPR 0.90-0.93 at 0.9-1.5 FP per case in the CV;
  across datasets TPR 0.81-0.87, precision 0.44-0.89.
- Acknowledged limitations (verbatim): rotation and downscaling can lose CMB; vessel removal can remove real
  CMB close to vessels; thresholds were set empirically per dataset; 5 mm slices create partial-volume problems;
  on T2*-GRE the contrast is weaker than on SWI.
- Paper against code (read myself): FRST per the paper averages over four radii 2/3/4/6 voxels -- in the code
  `radii = [2, 3]`. Post-processing per the paper: volume < 2.5 mm3 removed, ellipticity > 0.2 removed,
  < 5 mm from the mask edge removed; not yet checked in the code.
- 2025 (abstract only): not a new detection method, but size and location of CMB per MARS (infratentorial / deep /
  lobar) via atlases; the atlas of MARS regions in MNI space is public. Whether VALDO appears: not found.

## Answer
Based on what the paper states: MicrobleedNet used VALDO as data for tuning, but took nothing from the RESULTS of
the challenge -- it could hardly have done so given the timing. Conversely, none of the VALDO teams used FRST,
vessel removal, or distillation.

## What we take from both (interpreted)
| Lesson | From whom | In our work |
|---|---|---|
| Long training, large context, T1+T2+T2* | VALDO top teams | 60 -> 300 epochs running; patch size and T1/T2 channels are pending |
| 3D against mimics; 2D falls behind | VALDO | confirmed: our 2.5D network only rank 2, not statistically secured |
| Two-stage (candidates + classifier) is on par with the top | both | network 10 still to be built |
| Take anisotropy seriously | nnU-Net (VALDO winner) automatically chooses anisotropic kernels | our secured gain +0.16 |
| FRST as a guide channel | Sundaresan | keep; without inpainting 96% of CMB in the top 1% |
| Edge distance and shape filter as post-processing | Sundaresan | never measured in our work -> cheap grid search on existing maps |
| Rotation/scaling can destroy CMB | both warn | matches our negative augmentation result (-0.06) |
| Vessel removal can remove CMB | Sundaresan names it itself as a limitation | measured by us: 109 of 236 hit |
| Location of CMB per MARS with public atlas | Sundaresan 2025 | better than my SynthSeg approximation, which cannot separate deep white matter |
| Thresholds empirical per dataset | Sundaresan | in our work via the inner fold -- redetermine for catalina, do not carry over |
