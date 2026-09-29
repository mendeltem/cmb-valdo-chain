# Best strategy and replication -- as of 2026-09-23, 06:56

Purpose: ONE document that shows (1) what, based on all measurements to date, is the best strategy -- for VALDO and for
catalina --, (2) how we got there, and (3) with which commands and files every number can be recomputed. Justifications
are in `WORKLOG.md` (sections in parentheses), raw measurements in `protokoll.md`, private sums in
`/home/uchralt/data/work/mb-arena/protokoll.md`. catalina appears here only as sums; the data never leaves Alita.

All commands run in `/home/uchralt/data/work/valdo-t2s` with `PY=/home/uchralt/miniconda3/envs/dl/bin/python`.
Evaluations only need the CPU (`CUDA_VISIBLE_DEVICES="" nice ...`), trainings run via the queue
(`grossauftrag --einreihen --vram 20 --gib 14 -- <command>`), never two at the same time.

---

## 0 Summary: the best strategy

| | VALDO (T2*-GRE, public, 57 CV cases, 139 CMB) | catalina-T2* (private, 75 cases, 828 CMB) | catalina-SWI (private, 61 cases, 652 CMB) |
|---|---|---|---|
| Preprocessing | FSL FAST bias correction -> brain mask -> scale 1 - I/max (this is already a min-max 0-1 normalisation per case, see `WORKLOG.md` 11) -> **no** vessel inpainting -> grid 0.5 x 0.5 x 1.0 mm, brain-cropped -> FRST (radii 2, 3, 4, 6 voxels) as second channel | same chain; brain masks mostly supplied with the source data (generated there via HD-BET script, unchecked), 17 missing ones done ourselves with HD-BET | same chain |
| Network and training | anisotropic 3D U-Net `a03-aniso`, 60-epoch plan without early stopping (the best intermediate checkpoint of the inner fold is used, checked every 5 epochs), AdamW 3e-4 + cosine, BCE+Dice, 800 patches per epoch (38 x 62 x 94), 60 % lesion / 25 % brain / 15 % arbitrary, mirroring | **own training** with the same recipe OR fine-tuning the VALDO fold models (30 ep., lr 1e-4) -- equivalent; 120 ep. are confirmed worse (step 25); mixed with VALDO at 120 ep. on par with own 120, but -0.02 (n.s.) below own 60 | **own training**, NO start from the T2* network |
| Prediction | sliding window; mean of two seeds (reliably lifts an average model to the level of the best one); mirror TTA optional (+0.02) | one model; ensemble / TTA with no confirmed effect | one model |
| Post-processing (fixed) | threshold 0.3, >= 2 mm3, components > 2100 voxels removed, **CSF rule** (SynthSeg majority label) | same; CSF rule with SynthSeg `--robust` directly on T2* | same, but **WITHOUT the CSF rule**: measured 23 Sep, it costs 0.020 here (0.614 vs. 0.634) -- 74 real hemorrhages carry a CSF label |
| Second stage (since 21 Sep) | ONE small 3D-CNN for all cohorts on candidates from probability 0.15 upward (section 3a): on par with the CSF rule (0.625-0.636 vs. 0.627) | on par (0.672 vs. 0.670) | **confirmed better: 0.670 vs. 0.626 (+0.044 [+0.012; +0.085]), repeated with an independent first stage: 0.662 vs. 0.617 (+0.045 [+0.018; +0.085])**; the hard CSF rule HURTS here (74 real hemorrhages carry a CSF label) |
| Lesion F1 (pooled, out-of-fold) | 0.585 raw / **0.646** with CSF rule (seed 42); 0.553 / 0.590 (seed 1); 0.501 / 0.578 (seed 2); **mean across three seeds 0.546 raw / 0.605 with rule (range 0.07-0.08)** -- this is the figure to report for ONE model; two-seed ensemble 0.594 / **0.647** (shipped model), three members 0.581 / 0.636 | **0.663** raw / **0.675** with CSF rule; fine-tuning 0.657; without the three extreme cases 0.696 | **0.634** raw (with CSF rule 0.614 -- worse, see above); with shared second stage 0.670 |
| further metrics | VALDO rule median per subject 0.667; F1 per subject averaged 0.571 (only subjects with hemorrhage) resp. 0.626 (all) | sensitivity 0.66, precision 0.67, 3.6 false positives per case | sensitivity 0.65, precision 0.62, 4.3 false positives per case |

Not part of the best strategy, because measured without benefit: vessel inpainting (original and own), z-normalisation + augmentation,
narrow input mask, second stage PER COHORT (four classifiers; the SHARED second stage has been part of the chain since 21 Sep, 3a), lesion-wise loss, center map, mixing the cohorts at equal
budget, zero-shot. Details in section 5.

**Where we stand:** under the metric of the latest work by the VALDO organisers (CenSynCMB, arXiv 2607.05325: 74.3 with T1 + T2 + T2*)
we are at 0.57-0.63 -- within the range of their comparison methods (61-66). The gap lies in precision; our bottleneck is the
separation of hemorrhage from mimics with only ONE sequence (`WORKLOG.md` 5j, 5k).

---

## 1 Prerequisites

- Machine and software: `docs-umgebung/system.txt` (RTX A5000, i9-12900K, 62 GB; torch 2.6.0+cu124, monai 1.5.2, numpy 2.3.5, scipy 1.18.1,
  nibabel 5.4.2, scikit-learn 1.9.0; FreeSurfer 8.2.0; FSL 6.0.7.23); frozen environments `docs-umgebung/` (conda `dl`, `shiva`).
  HD-BET lives in the `medizin` environment. Never mix environments.
- VALDO 2021 Task 2 data: `/home/uchralt/data/extern/valdo2021/Task2` (72 cases, 236 CMB; license CC BY-NC-SA 4.0, `dev/licence-valdo.md`).
- catalina data: `~/data/sourcedata/catalina` (private). Reference original: MicrobleedNet commit 958f1cb; local copy `~/software/microbleed-detection`
  is our fork (`DEVIATIONS-microbleednet.md`, layer A).
- Code: `code/` (training scaffold, German, used unchanged), `cmb/` (clean English package: evaluation, transfer, second stage, QC).
  Version state: private repository `cmb-t2star-pipeline`, daily snapshot 23:40 (`github-cmb-sync.sh`); before 20 Sep only dated
  backup copies `*.vor-*`.

---

## 2 VALDO -- step by step

| No. | Step | Command | Result / check value |
|---|---|---|---|
| V1 | Split | `$PY code/split.py` (seed 0) | `dev/split.json`, `dev/cases.json`: 15 cases held out (97 CMB), 57 in 5 folds (139 CMB) |
| V2 | Bias correction + original chain | `$PY code/preprocess.py --out dev/preproc --all` | `dev/preproc/<id>/` with `fast_restore.nii.gz` (FSL FAST -B) |
| V3 | Input WITHOUT inpainting | `$PY code/preproc_roh_bauen.py` | `dev/preproc_roh/` (image = 1 - fast_restore / max within the mask) |
| V4 | Training grid | `GITTER_C_QUELLE=dev/preproc_roh GITTER_C_ZIEL=dev/gitter_d $PY code/gitter_c_bauen.py` | `dev/gitter_d/<id>/<id>_{image,frst,label}.nii.gz`, 72/72; component count source = target checked per case |
| V5 | Case list | `$PY -m cmb.transfer.build_valdo_manifest --grid dev/gitter_d --out dev/manifest_valdo_gitter_d.json` | 57 cases, 139 lesions; rebuilt on 20 Sep and compared with the file in use: identical |
| V6 | Tissue map | `$PY code/we5.py --synthseg` (mri_synthseg on T1 in T2* space, resampled onto the grid; 33 of 72 T1 carry NaN -> `nan_to_num` beforehand, otherwise SynthSeg silently returns an empty image) | `dev/synthseg/<id>_synthseg.nii.gz`, 72/72 |
| V7 | Training | `$PY code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d` (seed 42; second seed: `--seed 1`) | `ergebnisse/turnier/a03-aniso-e60-gd/fold<k>/`: `modell.pt`, `meta.json`, `verlauf.json`, `<id>_pred_proba.nii.gz` (out-of-fold); ~21 min per fold |
| V8 | Raw evaluation | `$PY -m cmb.transfer.evaluate --run basis=ergebnisse/turnier/a03-aniso-e60-gd --manifest dev/manifest_valdo_gitter_d.json` | F1 0.585, P 0.52, S 0.67, 1.51 false positives per case |
| V9 | Evaluation with CSF rule, fixed operating point | `$PY -m cmb.analysis.operating_point --run basis=... --manifest ... --synthseg "dev/synthseg/{id}_synthseg.nii.gz"` | fixed cell 0.3 / 2 mm3: **0.646** (cross-validated choice 0.622) |
| V10 | Ensemble of two seeds | `$PY -m cmb.analysis.ensemble --runs ergebnisse/turnier/a03-aniso-e60-gd ergebnisse/turnier/a03-aniso-s1-e60-gd --out ergebnisse/turnier/ens2-a03-aniso-e60-gd`, then V8/V9 | 0.594 raw, **0.647** with CSF rule, 0.82 false positives per case |
| V11 | Mirror TTA (GPU, ~25 min) | `$PY -m cmb.analysis.tta_predict --run ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --grid dev/gitter_d --out ergebnisse/turnier/a03-aniso-e60-gd-tta` | 0.604 raw (paired +0.019 [-0.017; +0.058]), 0.652 with CSF rule |
| V12 | Metric per subject (Challenge / CenSynCMB) | `$PY -m cmb.analysis.subject_level_f1 --run ... --manifest ... --synthseg ...` | median 0.667; mean 0.571 / 0.626 |
| V13 | Error pattern | `$PY -m cmb.analysis.fp_location ...`, `$PY -m cmb.analysis.missed_lesions ...` | false positives by location; sensitivity small / medium / large 0.52 / 0.69 / 0.79 |
| V14 | Review pages | `$PY -m cmb.review.build_qc ...`, `$PY -m cmb.review.build_review ...` | `qc.html` (two columns), candidate review page (201 candidates) |
| V15 | blinded review page for the label ceiling | `$PY -m cmb.review.build_review --blind --fixed-cell --repeat 20 --seed 0 --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --out ergebnisse/review/a03-aniso-e60-gd-blind --dataset VALDO --model a03-aniso-e60-gd`; evaluation `$PY -m cmb.review.summarise_review <Urteile>.json --key ergebnisse/review/a03-aniso-e60-gd-blind.key.json` | 54 false positives + 47 missed + 94 hit components (92 hemorrhages) + 20 repeats = 215 views; 2 x 92 / (2 x 92 + 54 + 47) = 0.646; rebuild bit-identical; judgment pending (`WORKLOG.md` 5b) |

Saved result files for these numbers: `ergebnisse/analysis/` -- `blob_vs_basis_valdo.json` (V8), `operating_point_valdo_members.json` (V9, all
individual models), `operating_point_valdo_ensembles.json` and `ensemble2_valdo.json` (V10), `tta_valdo.json`, `operating_point_valdo_tta.json` (V11),
`subject_level_f1_valdo.json` (V12), `fp_location_valdo.json`, `missed_lesions_valdo_blob.json` (V13).

Determinism: same seed -> same folds, same sample; demonstrated for a01 (four decimal places). Between seeds the
recipe scatters more than assumed up to 21 Sep: three seeds of the identical recipe 0.646 / 0.590 / 0.578 with CSF rule (raw 0.585 / 0.553 / 0.501), and seed 2 vs. 42 is
"confirmed" worse in the case bootstrap (-0.083 [-0.135; -0.033]). The bootstrap over cases does NOT contain the training noise. Rule since 22 Sep: an effect from one seed per side
counts as robust only from ~0.08 upward; smaller ones need a second seed or an independent first stage (`WORKLOG.md` 3, 5u).

---

## 3 catalina -- step by step (everything under `/home/uchralt/data/work/mb-arena`, private)

| No. | Step | Command | Result / check value |
|---|---|---|---|
| C1 | Brain mask + bias correction | `python hirn_und_bias.py t2star --alle` resp. `swi` (FSL `fast -B` for all; HD-BET from the `medizin` environment only for the 17 T2* cases without a supplied mask -- 58 T2* and 61 SWI masks come from the source) | `~/data/bids/catalina-<mod>/derivatives/{brainmask,biascorr}/` |
| C2 | Cohort and folds | `python kohorte_bauen.py` (seed 0; classes by CMB count 0 / 1-2 / 3-5 / > 5, each class distributed round-robin over 5 folds) | `faelle.json`, `participants.tsv`: T2* 75 cases, SWI 61; folds NOT to be changed anymore |
| C3 | Training grid | `$PY -m cmb.transfer.build_private_grids --modality t2star` (and `swi`) -- same functions as V4 | `gitter/<mod>/<id>/...`, `gitter/manifest_<mod>.json` (identifiers `c2s-` / `csw-`); 828 resp. 652 lesions on the grid |
| C4 | Tissue map T2* | SynthSeg 2.0 `--robust` directly on the T2* image, CPU (command and rationale: private protocol 19 Sep; without `--robust` 24 % of the CMB land in CSF) | `synthseg-t2star/robust/<id>_synthseg.nii.gz`, 75/75, 1 mm isotropic -- the tools resample it onto the grid themselves |
| C5 | own training | `$PY code/turnier.py --netz a03-aniso --epochen 60 --manifest <mb-arena>/gitter/manifest_t2star.json --name cat-t2s --basis <mb-arena>/transfer` (SWI: `manifest_swi.json`, `--name cat-swi`) | `transfer/a03-aniso-e60-cat-t2s/`, `.../a03-aniso-e60-cat-swi/` |
| C6 | fine-tuning | like C5 with `--epochen 30 --lr 1e-4 --init <valdo-t2s>/ergebnisse/turnier/a03-aniso-e60-gd/fold{fold}/modell.pt --name ft-t2s` | `transfer/a03-aniso-e30-lr0.0001-ft-t2s/` |
| C7 | zero-shot | `$PY -m cmb.transfer.zero_shot --models ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --manifest ... --out <mb-arena>/transfer/zero-shot-t2star` | mean of the five VALDO fold models |
| C8 | mixed | like C5 with `--manifest dev/manifest_valdo_gitter_d.json <mb-arena>/gitter/manifest_t2star.json --name mix-valdo-t2s` (all three: additionally `manifest_swi_mix.json`, `--name mix-alle`) | shared 5 folds across all cohorts |
| C9 | evaluation | `$PY -m cmb.transfer.evaluate --run nur-catalina=... feintuning=... zero-shot=... gemischt=... --manifest ...` (first run = reference for the paired comparisons) | see table below |
| C10 | CSF rule, error pattern, review page | `cmb.analysis.operating_point` / `missed_lesions` / `fp_location` with `--synthseg "<mb-arena>/synthseg-t2star/robust/{sid}_synthseg.nii.gz"`; `cmb.review.build_qc --out <mb-arena>/qc/...` | 0.675 with CSF rule; `qc/t2star-nur-catalina/qc.html` |

Results T2* (fixed post-processing, 75 cases, paired against own training; `transfer/auswertung_t2star_alle_0920.json` et al.):

| Strategy | F1 | Precision | Sensitivity | False positives per case | paired |
|---|---|---|---|---|---|
| SHIVA-CMB unchanged | 0.312 | 0.92 | 0.19 | 0.2 | -- |
| Zero-shot (VALDO only) | 0.622 | 0.60 | 0.65 | 4.8 | -0.041 [-0.069; -0.016] |
| mixed VALDO + catalina, 60 ep. | 0.623 | 0.56 | 0.70 | 6.1 | -0.040 [-0.077; -0.012] |
| all three cohorts mixed, 60 ep. | 0.618 | 0.58 | 0.66 | 5.3 | -0.044 [-0.082; -0.018] |
| Fine-tuning 10 ep. | 0.619 | 0.56 | 0.69 | 6.0 | -0.044 [-0.090; -0.003] |
| **Fine-tuning 30 ep.** | 0.657 | 0.63 | 0.69 | 4.4 | -0.006 [-0.029; +0.015] |
| **own training 60 ep.** | 0.663 | 0.67 | 0.66 | 3.6 | reference |
| own training + fine-tuning averaged | 0.669 | 0.66 | 0.68 | 3.9 | +0.007 [-0.007; +0.021] |
| lesion-wise loss | 0.634 | 0.58 | 0.70 | 5.7 | -0.029 [-0.059; -0.007] |
| mirror TTA | 0.663 | 0.67 | 0.65 | 3.5 | +0.000 [-0.012; +0.014] |

SWI (61 cases; `transfer/auswertung_swi_alle_0920.json`, `auswertung_swi_mixalle_0920.json`): own training **0.634**; fine-tuning from the T2* network 0.599
(-0.036 [-0.074; +0.003]); all cohorts mixed 0.547 (-0.087 [-0.126; -0.048]); zero-shot 0.305 (-0.330 [-0.441; -0.220]).

Reading warning (5i): three T2* cases carry 234 of the 828 hemorrhages and 145 of the 283 missed ones. Any breakdown of pooled numbers needs the
cross-check without these cases (`missed_lesions.py`, table 5); without them: sensitivity 0.77, F1 0.696. Whether the many 1-2-voxel markings in the cerebellum
of these cases are genuine hemorrhages is an open question for expert judgment.

---

## 3a Second stage: one classifier for all cohorts (`MB` = `/home/uchralt/data/work/mb-arena`; patches are image data -> private cohorts only under `$MB`)

| No. | Step | Command | Result / check value |
|---|---|---|---|
| S1 | Candidates VALDO (soft = without the hard CSF rule, tissue class as a field) | `$PY -m cmb.stage2.candidates --predictions ergebnisse/turnier/a03-aniso-e60-gd --grid dev/gitter_d --keep-csf --out ergebnisse/stage2/a03-aniso-e60-gd-weich` | 278 candidates (106 on a hemorrhage), reach 103 of 139 hemorrhages (sensitivity ceiling 0.74) |
| S2 | Candidates catalina-T2* | `$PY -m cmb.stage2.candidates --predictions $MB/transfer/a03-aniso-e60-cat-t2s --manifest $MB/gitter/manifest_t2star.json --synthseg "$MB/synthseg-t2star/robust/{sid}_synthseg.nii.gz" --keep-csf --out $MB/stage2/a03-aniso-e60-cat-t2s-weich` | 1082 candidates |
| S3 | Candidates catalina-SWI (seed 42 and, for confirmation, seed 1) | like S2 with `--predictions $MB/transfer/a03-aniso-e60-cat-swi` (resp. `a03-aniso-s1-e60-cat-swi`), `manifest_swi.json`, `synthseg-swi/robust`, `--out $MB/stage2/a03-aniso-e60-cat-swi-weich` (resp. `a03-aniso-s1-e60-cat-swi-weich`) | 870 candidates (464 on a hemorrhage), reach 461 of 652 (0.71) |
| S4 | shared 3D-CNN, nested 5-fold CV (GPU ~25 min) | `$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich swi=$MB/stage2/a03-aniso-e60-cat-swi-weich --folds swi=$MB/gitter/manifest_swi_mix.json --models cnn --skip-own --tissue --json $MB/stage2/pooled_weich_cnn_0921.json` (`--folds`: seven SWI cases are twins of T2* subjects and must go in their fold) | VALDO 0.625 / T2* 0.672 / SWI 0.670; paired against the plain threshold +0.059 [-0.002; +0.113] / +0.017 [+0.002; +0.041] / **+0.044 [+0.012; +0.085]**; against threshold + hard CSF rule -0.003 / +0.001 / +0.056 [+0.011; +0.103] |
| S5 | confirmation with independent first-stage SWI | like S4 with `swi=$MB/stage2/a03-aniso-s1-e60-cat-swi-weich`, `--json $MB/stage2/pooled_weich_cnn_swi_s1_0921.json` | SWI 0.662 vs. 0.617: **+0.045 [+0.018; +0.085]**; VALDO 0.636, T2* 0.672 |
| S6 | more context (queued 21 Sep, `WORKLOG.md` 5r) | S1-S3 additionally with `--half 32 32 16 --out ...-weich-h32`; then S4 with the `-h32` folders and `--models cnn cnn2s` | candidate rows identical to S1-S3, the center of the large patches is voxel-identical to the small ones (checked for all four sets); `$MB/stage2/pooled_h32_cnn_cnn2s_0921.json`: paired two-scale minus fine +0.003 / +0.016 [-0.002; +0.030] / +0.016 [-0.002; +0.037]; confirmation `..._swi_s1_0921.json`: **-0.030 [-0.059; -0.011]** / +0.019 [+0.001; +0.036] / +0.013 [-0.001; +0.031] -> not adopted (step 24); pairwise comparison: `$PY -m cmb.stage2.compare_runs --a <json> "pooled cnn2s" --b <json> "pooled cnn"` |

The commands S1-S3 were nowhere recorded verbatim until the evening of 21 Sep; they were reconstructed from the options and CHECKED for ALL FOUR candidate sets: the rebuild (with `--half 32 32 16`) yields the same rows (278 / 1082 / 870 / 893) and the same voxels at the center of the patches. S4 / S5 come verbatim from the
queue file (entries 137 and 144). The GPU does not compute bit-identically: two runs of the same `pooled cnn` differed by 0.005 on VALDO (0.625 / 0.630).

---

## 3b One-off measurement on the 15 held-out cases (2026-09-28; `ergebnisse/validierung/heldout_0928/`)

| No. | Step | Command | Result / check value |
|---|---|---|---|
| H1 | Manifest of the 15 cases (auxiliary fold 0) | `$PY -m cmb.transfer.build_valdo_manifest --grid dev/gitter_d --held-out --out dev/manifest_valdo_gitter_d_heldout.json` | 15 cases, 97 CMB |
| H2 | first stage per seed (GPU) | `$PY -m cmb.transfer.zero_shot --models ergebnisse/turnier/a03-aniso-e60-gd --net a03-aniso --manifest dev/manifest_valdo_gitter_d_heldout.json --out ergebnisse/validierung/heldout_0928/stufe1-s42` (and `a03-aniso-s1-e60-gd` -> `stufe1-s1`) | 2 x 15 maps, 377 s |
| H3 | two-seed ensemble | `$PY -m cmb.analysis.ensemble --runs .../stufe1-s42 .../stufe1-s1 --out .../stufe1-ens2` | 15 cases |
| H4 | evaluation | `cmb.transfer.evaluate` (raw), `cmb.analysis.operating_point --synthseg "dev/synthseg/{id}_synthseg.nii.gz"` (fixed cell), `cmb.analysis.subject_level_f1 --synthseg ...`, each with manifest H1 resp. `..._ohne110.json` / `..._nur110.json` | pooled 0.468 (CSF rule) / raw 0.463; without sub-110 0.731; median per case 0.667 |

## 3c Research line 5ae: synthesis and cubic grid (2026-09-28)

| No. | Step | Command | Result / check value |
|---|---|---|---|
| F1 | distribution of real hemorrhages | `$PY -m cmb.synthesis.lesion_stats --manifest dev/manifest_valdo_gitter_d.json --json ergebnisse/analysis/lesion_stats_valdo_0928.json` | 139 lesions, volume median 10 mm3, peak-contrast median 0.201 |
| F2 | synthesis grid | `$PY -m cmb.synthesis.build_synthetic_grid --manifest dev/manifest_valdo_gitter_d.json --stats ergebnisse/analysis/lesion_stats_valdo_0928.json --out dev/gitter_s --proc 8 --seed 0` | 57 cases, 209 synthetic CMB, 319 mimics, checks 57/57 (`dev/gitter_s/summary.json`) |
| F3 | cubic grid | `GITTER_C_QUELLE=$ROOT/dev/preproc_roh GITTER_C_ZIEL=$ROOT/dev/gitter_k GITTER_C_ORDNUNG=3 $PY code/gitter_c_bauen.py --faelle <57 CV-Ids> --proc 8` (linear beforehand checked bit-identical to gitter_d on sub-201) | 57 cases, label/mask bit-identical to gitter_d |
| F4 | trainings (queue 203-206) | `$PY code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_k [--seed 1]` and `... --gitter dev/gitter_d --synthese dev/gitter_s [--seed 1]` | folders `a03-aniso-e60-gk`, `a03-aniso-s1-e60-gk`, `a03-aniso-e60-gd-syn`, `a03-aniso-s1-e60-gd-syn` |
| F5 | Momeni/CSIRO grid (arm 3) | `$PY -m cmb.transfer.build_momeni_grid --processes 8` (source `~/data/extern/momeni-csiro-cmb`, locations from `rCMB_locations.json`) | 57 cases, 146 lesions, manifest `<extern>/gitter/manifest_momeni.json` |
| F6 | zero-shot on Momeni (queue 207-209) | `$PY -m cmb.transfer.zero_shot --models $MB/transfer/a03-aniso-e60-cat-swi --net a03-aniso --manifest <extern>/gitter/manifest_momeni.json --out <extern>/transfer/zero-shot-swi-s42` (likewise `a03-aniso-s1-e60-cat-swi`, `ergebnisse/turnier/a03-aniso-e60-gd`) | evaluation `cmb.analysis.subject_level_f1` + `cmb.transfer.evaluate` |
| F7 | arm 4: classifier VALDO -> catalina (queue 210-212) | `$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich swi=$MB/stage2/a03-aniso-e60-cat-swi-weich --folds swi=$MB/gitter/manifest_swi_mix.json --models cnn --skip-own --tissue --transfer valdo=t2star,swi --fractions 0 0.1 0.25 0.5 1 --seed-offset N --json $MB/stage2/transfer_valdo_N.json` (N = 0, 1, 2) | smoke test 1 epoch CPU ok (28 Sep 19:20); modes `transfer<fraction> cnn` against `pooled cnn` in the same run |
| F8 | arm 1, second version 5ae-1b (queue 213-214) | grid: `$PY -m cmb.synthesis.build_synthetic_grid --manifest dev/manifest_valdo_gitter_d.json --stats ergebnisse/analysis/lesion_stats_valdo_0928.json --out dev/gitter_s2 --proc 8 --seed 1 --profile flat --m-min 0 --m-max 0`; training: `$PY code/turnier.py --netz a03-aniso --epochen 60 --gitter dev/gitter_d --synthese dev/gitter_s2 --synthese-anteil 0.5 [--seed 1]` | 57 cases, checks 57/57, no mimics; folders `a03-aniso-e60-gd-syn50-s2`, `a03-aniso-s1-e60-gd-syn50-s2`; wrapper test: 62 real lesion patches bit-identical, 19 of 38 empty ones replaced |
| F9 | application mode: VALDO classifier on foreign candidates (29 Sep) | candidates: `$PY -m cmb.stage2.candidates --predictions <extern>/transfer/zero-shot-valdo-s42 --manifest <extern>/gitter/manifest_momeni.json --synthseg "<extern>/synthseg/{id}_synthseg.nii.gz" --keep-csf --out <extern>/stage2/zero-shot-valdo-s42-weich` (likewise swi-s42); application: `$PY -m cmb.stage2.apply --source ergebnisse/stage2/a03-aniso-e60-gd-weich --target momeni-valdo-unet=<extern>/stage2/zero-shot-valdo-s42-weich momeni-swi-unet=<extern>/stage2/zero-shot-swi-s42-weich --seeds 3 --json ergebnisse/analysis/5ae_0929/momeni_apply_valdo_classifier.json` (GPU, ~12 min) | threshold 0.50 from VALDO; Momeni F1 0.607 (VALDO U-Net) / 0.630 (SWI U-Net) |
| F10 | arm 5: vessel run-length/Frangi per candidate (CPU) and classifier with 32 mm z-context (queue 215-217) | `$PY -m cmb.analysis.vessel_run_length --run NAME=FOLDER --manifest M --synthseg PATTERN --json out.json --table tabelle.json`; candidates `cmb.stage2.candidates ... --keep-csf --half 20 20 20 --out ...-weich-z40` (rows identical to `-weich`, checked); `$PY -m cmb.stage2.pooled --cohort valdo=ergebnisse/stage2/a03-aniso-e60-gd-weich-z40 t2star=$MB/stage2/a03-aniso-e60-cat-t2s-weich-z40 swi=$MB/stage2/a03-aniso-e60-cat-swi-weich-z40 --folds swi=$MB/gitter/manifest_swi_mix.json --models cnn cnn-z32 --skip-own --tissue --seed-offset N --json $MB/stage2/z32_N.json` | run-length only Momeni +0.044; Frangi nowhere; z32 open |

## 4 Evaluation rules (apply to every number in this document)

- Ranking metric: pooled lesion F1, 26-connectivity components, touch rule (`code/metric.py::hits`). Always out-of-fold.
- ONE fixed post-processing for all runs and cohorts (0.3 / 2 mm3 / > 2100 voxels removed), chosen cross-validated on VALDO on 19 Sep and never
  adjusted per run since. The cross-validated re-selection per model (`operating_point.py`) is reported for information only -- it fluctuates over 57 cases by
  0.02-0.05 and is not fit as a basis for decisions.
- Comparisons always paired over the same cases, bootstrap over cases, 10,000 draws, 95 % interval (`evaluate.py`, first `--run` = reference).
- **The case bootstrap does not contain the training noise** (22 Sep, step 29): two runs of the identical recipe with a different seed differ by up to 0.08, and the interval marks that
  as "confirmed". Therefore: an effect from one seed per side counts as robust only from ~0.08 upward; smaller effects need a second seed (training) or an independent first stage
  (second stage, where training noise is ~0.03 on identical candidates, step 24). Individual findings below this threshold appear in section 5 as directional indications, not as proof.
- CSF rule: a component is dropped if its SynthSeg majority label is in {4, 5, 14, 15, 24, 43, 44}; parameter-free.
- Additionally for context: VALDO rule (`code/metric_valdo.py`: centroid < 5 mm, one-to-one, F1 per subject) via `subject_level_f1.py`.

---

## 5 The path: every decision with evidence

| No. | Step | Effect (VALDO CV, pooled F1) | Decision | Evidence |
|---|---|---|---|---|
| 1 | tournament with ten U-Net designs, 40 ep. | Attention U-Net 0.245; anisotropic U-Net **0.404** (best) | a03-aniso | `WORKLOG.md` 4.4; `code/turnier_rangliste.py` |
| 2 | early stopping -> fixed 60-epoch plan (intermediate checkpoint still chosen via the inner fold) | 0.404 -> 0.470; seed noise decreases | 60-ep. plan | 4.5, section 3, 11 |
| 3 | drop the original vessel inpainting | 0.470 -> **0.588**, paired +0.136 [+0.065; +0.226]; it affected 109 of 236 CMB | removed | 4.1 |
| 4 | own, gentle inpainting at three doses | 0.588 / 0.584 / 0.570 -- no gain anywhere | not adopted | 4.1 |
| 5 | CSF rule (SynthSeg) | 0.588 -> 0.612 | adopted | 4.3 |
| 6 | threshold 0.3 + minimum size 2 mm3, cross-validated | 0.633 (recomputed on 20 Sep with a coarser grid: fixed cell 0.646, cross-validated 0.622) | fixed cell | 5, 5j |
| 7 | bias correction / FRST channel / z = 1 mm each dropped individually | -0.022 / -0.053 / -0.036 | all three stay | 4.9, 4.11, 4.8 |
| 8 | z-normalisation + rotation augmentation; narrow SynthSeg input mask | -0.060; -0.048 | discarded | 4.7, 4.2 |
| 9 | second stage (threshold, logistic, forest, boosting, small 3D-CNN) | at best on par with the plain threshold -- even with 961-1135 catalina candidates | discarded | 5d, 5j |
| 10 | SHIVA-CMB as an external benchmark | VALDO 0.575 (knows 60 of 72 cases), catalina 0.312 | comparison only | 5a |
| 11 | transfer to catalina: four strategies, then epochs | own training = fine-tuning 30 ep.; 10 ep. too short; mixing and zero-shot confirmed worse | table in 3 | 5e, 5h |
| 12 | location and error analysis | no weak location; what is missed is what is small AND low-contrast; three extreme cases | reading warning | 5i |
| 13 | lesion-wise loss (blob loss) | VALDO: small CMB 0.52 -> 0.64, F1 -0.044 [-0.105; +0.003], second seed -0.063 [-0.129; +0.006]; catalina-T2* -0.029 confirmed | not adopted | 5j, 5m |
| 14 | center map as an auxiliary output (following CenSynCMB) | small CMB 0.52 -> 0.66, F1 -0.054 [-0.110; -0.006] | not adopted | 5k |
| 15 | ensemble over seeds; mirror TTA | ensemble ~ best member, +0.02-0.04 above the mean, flattens from two on; TTA VALDO +0.02, catalina +-0 | ensemble of two as the standard for VALDO | 5j |
| 15a | positive:negative ratio 1:2 (33 % lesion patches) | training collapses into the empty prediction (2 of 2 folds; test F1 0.09), run aborted | 60 % stays; 1:3 ... dropped | 5g |
| 16 | hard negatives (online mining of the model's own false positives) | +0.000 [-0.044; +0.048]; only effective on the training cases | not adopted | 5l |
| 17 | reweighting by misses (fnrw, following CenSynCMB) | -0.030 [-0.080; +0.019]; more sensitive, more false positives | not adopted | 5l |
| 18 | lesion-wise loss on catalina-SWI | -0.003 [-0.036; +0.024]; small CMB 0.37 -> 0.44 | not adopted | 5l |
| 19 | tissue map (SynthSeg) as five INPUT channels of the U-Net | VALDO -0.058 [-0.138; +0.013]; the network learns the CSF rule by itself (0 predictions in CSF), but otherwise makes more false positives | not adopted (catalina run pending) | 5m |
| 20 | ONE classifier for all cohorts, tissue class as a feature, candidates from more sensitive networks as extra training data (`cmb/stage2/pooled.py --tissue --extra ...`) | against the best simple chain per cohort: VALDO -0.009, T2* +0.011, **SWI +0.033 [+0.013; +0.057]** (0.626 -> 0.659; false positives per case 3.5 -> 2.9) | candidate for the final chain; confirmation with second SWI seed queued; 3D-CNN see step 22 | 5j |
| 21 | pretrain on all cohorts first, then fine-tune on the target cohort | T2* -0.039 [-0.083; -0.007], SWI -0.045 [-0.073; -0.021] against own training | discarded | 5h |
| 20a | confirmation attempt for step 20 with an independent SWI stage (seed 1) | forest + tissue feature + extra: +0.004 [-0.021; +0.041] against the plain threshold -- the SWI gain of the feature-based classifiers is NOT confirmed | second stage not part of the final chain for now; 3D-CNN confirmation queued | 5j |
| 20b | confirmation of the shared 3D-CNN with the same independent SWI stage | SWI 0.662 vs. 0.617, paired **+0.045 [+0.018; +0.085]** (with seed 42: +0.044) -- reproduced; VALDO +0.009, T2* +0.001 against the hard-rule chain | 3D-CNN becomes part of the final chain | 5j |
| 22 | shared 3D-CNN (earlier mistakenly called "3D-ResNet") as second stage (`pooled.py --models cnn`, with extra candidates) | VALDO 0.622 (on par), T2* 0.684 (+0.014 n.s.), **SWI 0.669 (+0.044 [+0.013; +0.084])** | candidate for the final chain | 5j |
| 23 | cross-check: fixed LAST checkpoint instead of choosing on the inner fold (`turnier.py --letzter-stand`, seeds 42 and 1) | the gap between seeds shrinks (0.032 -> 0.007 raw; 0.056 -> 0.020 with CSF rule), but the mean drops (0.569 -> 0.542; 0.618 -> 0.604); s42 paired -0.047 [-0.088; -0.010], s1 -0.007 [-0.068; +0.051]; two-seed ensemble 0.626 vs. 0.647 | checkpoint selection stays; the mean over seeds is reported, not the best run | 5o |
| 24 | second stage with more context: two-scale CNN (16 mm fine + 28 mm coarse) against the fine CNN on the same candidates, two runs (second with an independent SWI stage) | VALDO +0.003 / **-0.030 [-0.059; -0.011]**, T2* +0.016 / **+0.019 [+0.001; +0.036]**, SWI +0.016 / +0.013 (n.s.); on T2* the false positives drop at the same sensitivity; VALDO fluctuates by 0.03 between the runs on identical candidates = training noise of the second stage | not adopted; fine CNN stays | 5r |
| 25 | budget question on mixing: own catalina-T2* training with 120 ep. as a control against mixed 120 ep. | own 120 ep. 0.640 vs. own 60 ep. 0.663: **-0.022 [-0.052; -0.002]** (false positives 3.6 -> 5.0); mixed 120 vs. own 120: +0.001 [-0.021; +0.023] (pre-registered comparison); mixed 120 vs. own 60: -0.021 [-0.054; +0.006] | at equal budget per case, mixing no longer costs anything, but it also brings nothing; the best strategy remains own training 60 ep.; budget per case (~1000-1400 patches) is the right quantity, not the epoch count | 5h |
| 26 | "120 epochs generally?": base recipe on VALDO with 120 ep. (prediction made in advance: no gain) | 0.556 vs. 0.585 raw (-0.029 [-0.068; +0.019]), 0.610 vs. 0.646 with CSF rule; false positives 1.51 -> 1.84; cohort 0.8 mm confirmed worse (-0.046) | no -- the right quantity is patches per training case (~1000-1400); 120 ep. only with correspondingly more cases | 5h |
| 27 | patch size, small (24 x 48 x 48 voxels = 24 mm cube, close to Sundaresan's 48^3) | 0.369 vs. 0.585 raw, **-0.216 [-0.277; -0.156]**; with CSF rule 0.491 vs. 0.646; S 0.74, but 5.5 false positives per case; all three cohorts negative | discarded -- for a single-stage network, 24 mm of context is too little; large patch 64 mm is running | 5t |
| 28 | patch size, large (48 x 96 x 96 voxels = 48 mm cube, 2x the voxels of the recipe; 64 mm not measurable: cgroup memory limit) | 0.569 vs. 0.585 raw (-0.016 [-0.074; +0.034]), 0.609 vs. 0.646 with CSF rule; per cohort all n.s.; 29 instead of 21 min per fold | not adopted; dose series 24 / recipe / 48 mm: 0.369 / 0.585 / 0.569 -- the recipe sits at the optimum | 5t |
| 29 | third seed of the base recipe (`--seed 2`) and three-seed ensemble | s2 0.501 raw / 0.578 with rule; against s42 **-0.083 [-0.135; -0.033]** (identical recipe!); ensemble of 3: 0.636 vs. ensemble of 2: 0.647 | noise level raised to ~0.08 (section 4 here, `WORKLOG.md` 3); report the base as the mean 0.605 +- 0.034; two-seed ensemble remains the standard | 5u |
| 30 | thick-slice simulation as augmentation (`--aug-schicht`; hypothesis: gain in the thick cohorts) | 0.458 vs. 0.585 raw, **-0.127 [-0.179; -0.075]** (above the noise level); per cohort -0.045 / -0.125 / -0.146 -- the biggest loss exactly where the gain was predicted; 3.3 false positives per case | discarded; lesson: augmentation must match both the image AND the label statistics of the target; zero-shot on catalina still running (main metric 5p) | 5p |
| 31 | intensity augmentation (`--aug-intensitaet`: gamma, bias field, contrast, brightness, noise, each p = 0.5; expectation: even result) | 0.477 vs. 0.585 raw, **-0.108 [-0.180; -0.030]**; S 0.56 AND P 0.41 both drop; below the weakest seed (0.501) | discarded within VALDO (too strong a perturbation for the budget); zero-shot on catalina running (main metric 5p); robustness probe (5v) checks whether the base is already invariant | 5p, 5v |
| 32 | augmentation, main metric: zero-shot on catalina (noise level = zero-shot of the base with seed 1) | T2*: s1 +0.016 [+0.003; +0.033], intensity -0.014, thick-slice -0.016; SWI: s1 +0.006, intensity -0.007, thick-slice **-0.065 [-0.096; -0.036]** (16 false positives per case); MORE false positives everywhere instead of fewer | both augmentations removed; side finding: the seed that is weaker in the CV transfers better -> ship the two-seed ensemble, never "the best seed" | 5p |
| 33 | architecture cross-check on clean data, part 1: nnU-Net design `a02-nnunet` in today's recipe (60 ep., gitter_d) | 0.458 vs. 0.585 raw, **-0.127 [-0.202; -0.048]**; 0.515 vs. 0.646 with CSF rule; behind on all five folds and all three cohorts; S AND P lower | tournament ranking holds; base stays; Attention U-Net running | 5s |
| 34 | architecture cross-check, part 2: Attention U-Net `a01-attunet` in today's recipe | 0.463 vs. 0.585 raw, **-0.121 [-0.190; -0.061]**; 0.536 vs. 0.646 with CSF rule; S 0.68 like the base, but P 0.35 and 3.1 false positives per case; cohort 3-4 mm **-0.200** | tournament ranking confirmed on clean data (2 of 2 challengers); base stays final | 5s |
| 35 | user question: can the second stage turn the Attention U-Net's high sensitivity into F1? What is measured is the CEILING of the candidate set (`cmb.stage2.candidates --keep-csf`, CPU) | hemorrhages reached out of 139: base 103, attunet **104** (with 407 instead of 278 candidates), blob 110, nnunet 92; F1 ceiling with an error-free second stage 0.851 / 0.856 / 0.884 / 0.797 | no, no GPU run; rule: a first stage only helps the chain via a higher CEILING, not via sensitivity at the operating point (counter-test blob: ceiling 0.884, chain nonetheless 0.594 vs. 0.625) | 5s, 5j |

Running or queued (as of 22 Sep 20:27; steps 23-35 are done -- training recipe measured, augmentation negative on all three counts, architecture choice confirmed on clean data): only the robustness probe on the finished network remains
(`cmb.analysis.robustness_probe`, ~2 GPU-h, 5v) and the robustness probe on the finished network (`cmb.analysis.robustness_probe`, ~2 GPU-h; procedure and expectation set out in advance in `WORKLOG.md` 5v). The zero-shot runs of the
augmentation (`cmb.transfer.zero_shot --models ergebnisse/turnier/a03-aniso-e60-gd-augi|-augs|a03-aniso-s1-e60-gd`) have been evaluated (step 32). Dropped on 21 Sep (entries remain with justification in the queue file): ratio 1:1 + chain, attention on the winner, percentile scale seed 1, fine-tuning 60 ep. (T2*, SWI), tissue channel on catalina.
Every training command is recorded with a timestamp and return value in `/home/uchralt/local_agentic_system/system/logs/grossauftrag-schlange.jsonl`. The numbers in square brackets are the display numbers AT THE TIME OF QUEUING; every reordering shifts them (on 21 Sep 21:52 by 2) -- what counts is the command, not the number.

---

## 6 Limits of replicability (named honestly)

1. Until 20 Sep there was no version control; the code state of older runs can only be reconstructed via dated backup copies (`*.vor-*`) and `meta.json` per fold.
   Since 20 Sep, a daily snapshot in the private repository.
2. Evaluations before 20 Sep partly ran as one-off scripts within the session. As of 21 Sep: ALL exist as files and have been checked against the logged numbers:

   | Evaluation | File / command | logged | from the file | verdict |
   |---|---|---|---|---|
   | Damage tally of the original inpainting | `$PY code/uebermalung2.py --pruefen dev/preproc` | 109 / 61 / 27 / 49 / 17 of 236; 12.2 / 11.7 / 17.3 % of the brain | identical | exact |
   | Mask volume, non-brain fraction | `$PY -m cmb.analysis.mask_volume --grid dev/gitter_d --synthseg "dev/synthseg/{id}_synthseg.nii.gz"` | 1419 / 1770 / 2533 ml (2085-3009); 4 / 16 / 40 % | identical | exact |
   | Cohort breakdown (19 Sep 07:00) | `$PY -m cmb.analysis.cohort_breakdown --run ergebnisse/turnier/a03-aniso ergebnisse/turnier/a01-attunet ...` | a03-aniso 0.500 / 0.359 / 0.471 (with CSF rule 0.510 / 0.478 / 0.490) | identical | exact |
   | FRST hit rate on the CMB | `$PY -m cmb.analysis.frst_hit_rate --grid dev/gitter_c dev/gitter_d` | 201/236 = 85 % -> 227/236 = 96 % | **194/236 = 82 % -> 226/236 = 96 %** | deviates (7 resp. 1 hemorrhage); the file computes the same way as `gitter_c_bauen.py::frst_guete` and exactly matches its build-time values for the 57 CV cases (124/139, 135/139); the one-off code from 19 Sep is not preserved. From now on 82 % / 96 % apply; the statement (inpainting weakens the FRST in the thick-slice cohorts: cohort 1 78 -> 99 of 106, cohort 3 26 -> 34 of 34) stands. |
   | Contrast loss from the resampling | `$PY -m cmb.analysis.resampling_contrast --source dev/preproc_roh --grid dev/gitter_d --manifest dev/manifest_valdo_gitter_d.json` | -23 / -12 / -32 % (peak -20 / -11 / -22 %) | **-26 / -15 / -30 %** (peak -17 / -13 / -16 %), volume -1 / 0 / 0 % | deviates in magnitude, same pattern; the ring definition is now given in the docstring (0.5-2.0 mm within the slice, inside the brain, excluding other hemorrhages). From now on the values from the file apply. |

   In addition since 20/21 Sep: paired comparison and cohort switch (`evaluate.py --split-valdo`), operating point (`operating_point.py`), location of errors (`fp_location.py`), misses
   (`missed_lesions.py`), metric per subject (`subject_level_f1.py`), case list (`build_valdo_manifest.py`), ensemble, mirror TTA. Current model per cohort (fixed cell, raw):
   cohort 1 (4 mm) 0.557, cohort 2 (0.8 mm) 0.634, cohort 3 (3-4 mm) 0.479 (`ergebnisse/analysis/basis_je_kohorte_valdo.json`).
3. Step 6 of the ladder: the 0.633 reported on 19 Sep comes from a one-off grid search with a different raster; today's tool gives 0.646 (fixed
   cell) and 0.622 (cross-validated). For a publication, today's computation, available as a file, applies.
4. The 15 held-out VALDO cases were used once on 14 Sep; all numbers here come from the 5-fold CV of the remaining 57. The one final measurement
   is still pending and must be disclosed.
5. Seed scatter: for the base recipe there are two real seeds (42, 1; a third is queued), for catalina one per strategy. Statements about
   catalina strategies rest on paired intervals over cases, not over seeds.
6. catalina is private: steps C1-C10 can only be reproduced on Alita; only sums go outward.
7. Literature figures in `research/` are mostly from search snippets and marked "unverified"; checked at the source is CenSynCMB (abstract and
   full-text HTML, 20 Sep). The errors in the original (A1-A3) were read from the source text, not executed.
