# Traceability: every path and every source

This page is the map. For every number in the README it names the data it was computed on, the command that produced it,
the file it was written to and the section of the working documents that records the decision. For every building block
of the chain it names where the idea comes from (paper, code, or own measurement) and which file in this repository
implements it. Nothing here is new; it only links what `docs/WORKLOG.md` (the lab book), `docs/REPLICATION.md`
(the commands), `docs/OVERVIEW-tests-2026-09-29.md` (every test in one table), `docs/DEVIATIONS-microbleednet.md`
(differences to the original) and `docs/LITERATURE.md` (the bibliography with a checked / unchecked flag per entry) contain.

Reading rule: `WORKLOG 4.1` means section 4.1 of `docs/WORKLOG.md`; `V9`, `C5`, `S4` are the step numbers of
`docs/REPLICATION.md`; `kern: csf` is the step name in `cmb/experiments/kern.py` (`python -m cmb.experiments.kern --list`
prints every step with its done-check and its exact command).

## 1 Where the data comes from

| Data | Origin | Licence | What we use | Where it lives on the original machine |
|---|---|---|---|---|
| VALDO Task 2 (72 T2*-GRE cases, 3 cohorts, 236 reference CMB) | Sudre et al., "Where is VALDO?", Medical Image Analysis 2024; download Zenodo 10.5281/zenodo.4520773 | CC BY-NC-SA 4.0 (`docs/licence-valdo.md`) | 57 cases in 5 folds for everything (`dev/split.json`, seed 0, `code/split.py`); 15 held out and scored once on 2026-09-28 (`WORKLOG 5ad`) | `~/data/extern/valdo2021/Task2/sub-XXX/` |
| Momeni / CSIRO SWI (57 cases with 146 reference points) | Momeni et al., CMPB 2021; data DOI 10.25919/AEGY-NY12 | CSIRO Data Licence, non-commercial, no redistribution | external validation only, no training (`WORKLOG 5ae`, `kern: momeni-*`) | `~/data/extern/momeni/` |
| In-house T2* (75 cases, 828 CMB) and SWI (61 cases, 652 CMB) | clinical BIDS cohort, reference masks by the clinic | never leaves the clinic | own training per modality, candidates for the shared classifier; only sums appear in this repository | `~/data/work/mb-arena/` (`REPLICATION 3`) |
| Brain masks | VALDO: supplied with the data (cohort 3 is a head mask, `WORKLOG 4.2`); in-house: supplied, HD-BET for 17 T2* cases | -- | mask for the intensity scaling and the metric | `dev/preproc/`, `derivatives/brainmask/` |
| Tissue maps | FreeSurfer 8.2 `mri_synthseg` (VALDO: on the T1 in T2* space; in-house and Momeni: `--robust` directly on the T2*/SWI) | FreeSurfer licence | CSF rule, tissue feature of the classifier, location of errors | `dev/synthseg/`, `synthseg-*/robust/` |

Nothing in `dev/` is image data: `cases.json`, `split.json`, the manifests and `landmarken_valdo.json` are lists and numbers.

## 2 Where the code comes from

| Part | Origin | In this repository | Differences recorded in |
|---|---|---|---|
| MicrobleedNet (reference method) | Sundaresan et al., Front. Neuroinform. 2023; code github.com/v-sundaresan/microbleed-detection, commit 958f1cb | re-implemented, then changed piece by piece; the original itself is installed as a package and used for FRST and as the fine-tuned baseline (`cmb/baselines/microbleednet_finetune.py`) | `docs/DEVIATIONS-microbleednet.md` (every deviation with the measurement that justified it); errors found in the original: `WORKLOG` A1-A3 |
| FRST channel | Loy & Zelinsky, IEEE TPAMI 2003, as used by MicrobleedNet | `code/frst.py` (self-test on synthetic discs and lines, `pipeline/00_selftest.py`) | `WORKLOG 4.11` (-0.053 without it) |
| Vessel inpainting of the original | Frangi et al. 1998 as applied by MicrobleedNet | removed; its damage tallied by `code/uebermalung2.py --pruefen` (109 of 236 reference CMB painted over) | `WORKLOG 4.1`, `REPLICATION 5` step 3 and 4 |
| Networks of the tournament | U-Net / 3D U-Net, Attention U-Net, UNet++, SegResNet, Swin UNETR, ResNet-SE, nnU-Net design (sources in `LITERATURE.md` tier C) | `code/netze.py` (all), `code/attention_unet.py`; winner `a03-aniso` also in `cmb/models/anisotropic_unet.py` with a bit-identity test against the old code (`cmb/tests/test_models_match_legacy.py`) | `WORKLOG 4.4` (tournament), `5s` (re-measured on clean data: -0.127 / -0.121 against the winner) |
| Training recipe | nnU-Net (anisotropic first level, deep supervision), AdamW + cosine, InstanceNorm, BCE + Dice | `code/turnier.py`, `code/cv5.py` | `WORKLOG 4.5`, `5o` (fixed epoch count), `5u` (seed noise 0.08) |
| Lesion metric | VALDO Task 2 official rule (26-connectivity, touch rule), evaluation code github.com/WhereIsValdo/valdo-eval-2021 | `code/metric.py` (self-test), `code/metric_valdo.py` (official rule), `cmb/metrics.py` with `cmb/tests/test_metrics_match_legacy.py` | `WORKLOG 2a`, `REPLICATION 4` (evaluation rules) |
| CSF rule | own; tissue map from SynthSeg (Billot et al. 2023) | `cmb/analysis/operating_point.py` (`--synthseg`) | `WORKLOG 4.3` (98 of 176 false positives removed, no true positive lost) |
| Second stage | own; idea of false-positive reduction on centred candidates after Setio et al. 2016 and Dou et al. 2016 | `cmb/stage2/candidates.py`, `cmb/stage2/pooled.py`, `cmb/stage2/apply.py` | `WORKLOG 5d` (per-cohort: nothing), `5j`, `3a` (shared: +0.045 SWI), `5w` (ceiling) |
| Baselines | SHIVA-CMB (Tsuchida et al. 2024), nnU-Net v2 (Isensee et al. 2021), fine-tuned MicrobleedNet | `cmb/baselines/shiva_cmb.py`, `cmb/baselines/nnunet_v2.py`, `cmb/baselines/microbleednet_finetune.py` | `WORKLOG 5a` (SHIVA 0.312 in-house), arm 6 and arm 10 of `5ae` |
| Ideas tested and rejected | blob loss (Kofler et al. IPMI 2023), centre map and miss-reweighting (CenSynCMB, He et al. 2026), hard negatives (OHEM), landmark alignment (Nyul-Udupa), synthetic lesions (Momeni et al. 2021) | `code/turnier.py --verlust blob`, `--zentrum`, `--fnrw`, `--ohem`; `code/landmarken_bauen.py`; `cmb/synthesis/` | `WORKLOG 5j, 5k, 5l, 5ac, 5ae`; one line each in `OVERVIEW-tests-2026-09-29.md` |

Every entry of `docs/LITERATURE.md` carries a flag: **Q** checked at the source, **R** from a research note (snippet or abstract),
**G** standard reference from memory. Twenty entries were cross-checked against the sources on 2026-09-23
(`docs/research/literature-check-2026-09-23.md`); cite nothing with flag R or G without opening it.

## 3 The path from the raw image to the deliverable (VALDO)

Every row is one step of `cmb/experiments/kern.py`; the pipeline scripts `pipeline/01_*.py` to `pipeline/11_*.py` call the
same steps in the same order. Inputs and outputs are relative to the repository root on the original machine.

| Step (`kern`) | Reads | Command (`REPLICATION` step) | Writes | Establishes | Recorded in |
|---|---|---|---|---|---|
| preprocess | `~/data/extern/valdo2021/Task2/` | `code/preprocess.py --all --out dev/preproc --skip-done` (V2) | `dev/preproc/<id>/fast_restore.nii.gz` | FSL FAST -B once per case; never repeated | `WORKLOG 4.9` |
| preproc-roh | `dev/preproc/` | `code/preproc_roh_bauen.py` (V3) | `dev/preproc_roh/<id>/<id>_image.nii.gz` | inverted image WITHOUT vessel inpainting: +0.136 | `WORKLOG 4.1` |
| grid | `dev/preproc_roh/` | `GITTER_C_QUELLE=... GITTER_C_ZIEL=dev/gitter_d code/gitter_c_bauen.py --proc 8` (V4) | `dev/gitter_d/<id>/<id>_{image,frst,label}.nii.gz`, `info.json` | 0.5 x 0.5 x 1 mm grid, FRST channel; component count source = target checked per case | `WORKLOG 4.8, 4.11` |
| manifest | `dev/gitter_d/`, `dev/split.json` | `cmb.transfer.build_valdo_manifest` (V5) | `dev/manifest_valdo_gitter_d.json` | 57 CV cases, 139 lesions; held-out 15 excluded | `WORKLOG 5ad` |
| synthseg | T1 of VALDO (33 of 72 carry NaN, `nan_to_num` first) | `code/we5.py --synthseg --parallel 4` (V6) | `dev/synthseg/<id>_synthseg.nii.gz` | tissue map on the grid for the CSF rule | `WORKLOG 4.3` |
| train-basis, train-s1 | `dev/gitter_d/`, manifest | `code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d [--seed 1]` (V7) | `ergebnisse/turnier/a03-aniso-e60-gd/fold<k>/{modell.pt,meta.json,verlauf.json,<id>_pred_proba.nii.gz}`, same with `-s1-` | raw 0.585 (seed 42), 0.553 (seed 1) | `WORKLOG 4.5, 5u`; every training command with timestamp and return value in the queue log `grossauftrag-schlange.jsonl` |
| ensemble | the two run folders | `cmb.analysis.ensemble --runs ... --out ergebnisse/turnier/ens2-a03-aniso-e60-gd` (V10) | `ens2-.../fold*/<id>_pred_proba.nii.gz` | mean of two seeds | `WORKLOG 5j` |
| evaluate-basis | run folders, manifest | `cmb.transfer.evaluate --run basis=... s1=... ens2=...` (V8) | stdout, `ergebnisse/analysis/blob_vs_basis_valdo.json` | raw lesion F1 at the fixed cell 0.3 / 2 mm3 | `REPLICATION 4` |
| csf | run folders, manifest, `dev/synthseg/` | `cmb.analysis.operating_point --run ... --synthseg "dev/synthseg/{id}_synthseg.nii.gz"` (V9) | `ergebnisse/analysis/operating_point_valdo_members.json`, `operating_point_valdo_ensembles.json`, `ensemble2_valdo.json` | **0.647** with CSF rule (ensemble); 0.646 / 0.590 single seeds | `WORKLOG 4.3, 5`; `REPLICATION 5` step 5 and 6 |
| candidates-valdo | `ergebnisse/turnier/a03-aniso-e60-gd`, grid | `cmb.stage2.candidates --keep-csf --out ergebnisse/stage2/a03-aniso-e60-gd-weich` (S1) | `patches.npy` and the candidate table | 278 candidates at 0.15 / 1 mm3 | `WORKLOG 3a` |
| stage2-pooled, stage2-transfer | candidate sets of all three cohorts | `cmb.stage2.pooled --cohort valdo=... t2star=... swi=... --models cnn --skip-own --tissue` (S4, S5) | `mb-arena/stage2/pooled_weich_cnn_*.json` | shared 3D CNN: SWI +0.045 [+0.018; +0.085]; VALDO-only classifier transfers zero-shot | `WORKLOG 3a`, `REPLICATION 5` steps 20b, 22 |
| heldout | the frozen ten model files (`ergebnisse/validierung/heldout_0928/FROZEN.sha256`) | run once on 2026-09-28; the step only prints a reminder | `ergebnisse/validierung/heldout_0928/summary.json` | 0.468 pooled, 0.731 without sub-110, median 0.667 | `WORKLOG 5ad` (rules set in advance) |
| momeni-grid, momeni-zeroshot, momeni-synthseg, momeni-score | `~/data/extern/momeni/` | `cmb.transfer.build_momeni_grid`, `cmb.transfer.zero_shot`, `cmb.transfer.momeni_synthseg`, `cmb.analysis.operating_point` | `<momeni>/gitter/manifest_momeni.json`, `transfer/zero-shot-valdo-s42/zero_shot.json`, `synthseg/` | external validation 0.613 with CSF rule, 0.630 with the SWI U-Net and the VALDO classifier | `WORKLOG 5ae` arm 4 |
| mbnet-prepare, mbnet-finetune, mbnet-score | VALDO raw data, our folds | `cmb.baselines.microbleednet_finetune prepare / finetune --fold k / score --cohort valdo` | `mbnet/valdo/falten/`, `mbnet/valdo/ergebnisse/fold*/lauf.json`, logs `logs/arm6_mbnet_score_*_0929.log` | fair baseline, same cases and metric: VALDO 0.040 native, in-house SWI 0.406, T2* 0.277 | `WORKLOG 5ae` arm 6 |
| qc-viewer | ensemble run, manifest, tissue maps | `cmb.review.build_slice_viewer --dataset VALDO ... --out ergebnisse/qc/valdo_slices` | `quality_control.html`, `data.json`, `img/` | the pages at mendeltem.github.io/valdo-cmb-qc/ | `WORKLOG 8` |

The in-house path (C1-C10) and the second stage (S1-S6) are tabulated the same way in `docs/REPLICATION.md` sections 3 and 3a;
the scripts for it are `catalina/*.py`, `cmb/transfer/build_private_grids.py` and the same training and evaluation modules.

## 4 Every number in the README, traced

| Number (README "Results") | Data | Produced by | Result file | Decision / discussion |
|---|---|---|---|---|
| MicrobleedNet re-implementation on VALDO: **0.245** | 57 CV cases | tournament of ten designs, 40 epochs, `code/turnier.py`, ranking `code/turnier_rangliste.py` | `ergebnisse/turnier/a01-attunet-*` | `WORKLOG 4.4`, `REPLICATION 5` step 1 |
| our chain, VALDO 5-fold CV, two-seed ensemble + CSF rule: **0.647** | 57 CV cases | V7, V10, V9 | `ergebnisse/analysis/operating_point_valdo_ensembles.json` | `REPLICATION 5` steps 2-7 (the ladder 0.245 -> 0.404 -> 0.470 -> 0.588 -> 0.612 -> 0.647) |
| held-out 15 cases: **0.468 pooled / 0.731 without one case / median 0.667** | 15 held-out cases, scored once | frozen chain, 2026-09-28 | `ergebnisse/validierung/heldout_0928/summary.json`, `FROZEN.sha256` | `WORKLOG 5ad` |
| in-house T2* / SWI, own U-Net + shared classifier: **0.671 / 0.672** | 75 / 61 private cases | C5 (own training 0.663 / 0.626), S4-S5 (shared classifier) | `mb-arena/transfer/auswertung_t2star_alle_0920.json`, `auswertung_swi_alle_0920.json`, `mb-arena/stage2/pooled_weich_cnn_swi_s1_0921.json` | `REPLICATION 3, 3a`; `WORKLOG 5e, 5h, 3a` |
| fine-tuned original on SWI: **0.406** (T2* 0.277, VALDO 0.040) | same folds | `cmb.baselines.microbleednet_finetune` | `logs/arm6_mbnet_score_{valdo,t2star}_0929.log` | `WORKLOG 5ae` arm 6 (its second stage did not learn) |
| external SWI cohort, no adaptation: **0.613 / 0.630** | Momeni 57 cases | `kern: momeni-*` | `<momeni>/transfer/zero-shot-valdo-s42/zero_shot.json` | `WORKLOG 5ae` arm 4 |
| seed noise of one training: **~0.08** | three seeds of the identical recipe | V7 with `--seed 1`, `--seed 2` | `ergebnisse/analysis/valdo_seeds_s2_gepoolt*.json` | `WORKLOG 5u`; rule in `REPLICATION 2` |
| "what did not help" (losses, augmentation, epochs, patch size, harmonisation, synthesis, second-stage variants) | VALDO CV, paired bootstrap | one row each in `REPLICATION 5` steps 8-35 | `ergebnisse/analysis/*.json`, `ergebnisse/turnier/<variant>/` | `OVERVIEW-tests-2026-09-29.md` (verdict per lever), `WORKLOG 5j-5ae` |

Evaluation rules that apply to every number (26-connectivity, touch rule, fixed cell chosen on the inner folds, paired case
bootstrap with 10 000 draws, "confirmed" = the 95 % interval excludes zero): `docs/REPLICATION.md` section 4.

## 5 What is written down automatically

- Per training run and fold: `meta.json` (every recipe parameter, seed, chosen epoch, threshold, duration, VRAM),
  `verlauf.json` (loss and validation per epoch), `modell.pt`, probability maps per case.
- Per evaluation: a JSON under `ergebnisse/analysis/` or `mb-arena/transfer/auswertung_*.json`; the operating-point and ensemble
  tools write their inputs (run names, manifest, synthseg pattern) into the file.
- Every GPU job: the queue log `grossauftrag-schlange.jsonl` with command, timestamp and return value (private machine).
- The environment: `env/environment-from-history.yml`, `env/requirements-lock.txt`, `docs/environment/` (conda `dl` and `shiva`,
  hardware, FreeSurfer 8.2.0, FSL 6.0.7.23).
- Backup copies of every changed script before 2026-09-20 with the date in the file name (`*.vor-*`), since then git.
- Checksums of the frozen deliverable: `ergebnisse/validierung/heldout_0928/FROZEN.sha256`.

## 6 Known gaps (do not hide them)

Listed in full in `docs/REPLICATION.md` section 6 and `docs/WORKLOG.md` section 10: no version control before 2026-09-20;
two early one-off analyses (FRST hit rate, contrast loss) do not reproduce to the digit and the values from the files apply;
the 15 held-out cases were touched twice (2026-09-14 and the final 2026-09-28); GPU training is not bit-identical between
runs (0.005 on the second stage); the in-house cohort can only be reproduced on the original machine; literature flagged R or G
is unchecked.
