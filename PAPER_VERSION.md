# Paper version: what this tag is, and every change against the code that ran

Tag `paper-v3-2026-11` is the code that produced the results of the clinical CMB paper, manuscript version v3
("How many local scans does a public microbleed detector need?", working title). The research code was never a git
repository; this file records how it was brought here and what was changed.

## Where the code came from

| repository path | source (research work folder, state of 05.-12.11.2026) | used by |
|---|---|---|
| `cmb/` | the research package `cmb` (last change 05.11.2026; MARS majority fix 11.11.) | every phase |
| `code/` | the training and grid scripts (`turnier.py`, `cv5.py`, `netze.py`, `gitter_c_bauen.py`, `frst.py`, ...) | stage 1 in every phase |
| `scripts/paper/phase1_arena/` | the arena and pooling job scripts (wc3-wc7, wk0, wk1/wk1b, wk2/wk2b, wk9) | Phase 1 |
| `scripts/paper/phase2_finetune/` | the fine-tuning blocks, calibration-only arm and frozen-V chains (wk0b, wk4b) | Phase 2 |
| `scripts/paper/phase3_cohorts/` | patient-level counts, MARS, site-count curve (wk3, wk5, wk5b) | Phase 3 |
| `scripts/paper/manuscript_v3/` | the v3 recomputation (wk11a-f), number check, figure and table scripts | manuscript v3 |
| `scripts/prep_bids/` | brain mask + bias correction + SynthSeg of the in-house BIDS cohorts (renamed from a folder named after the cohort) | preprocessing |

`scripts/paper/README.md` maps every script to its job and to the manuscript section, table or figure it feeds.

## Changes for portability and privacy (the logic is unchanged)

1. **Paths.** Every absolute path of the original machine became a name in `cmb/paths.py` (`ROOT`, `PRIVATE`, `VALDO`,
   `MOMENI`, `VEIN_ATLAS`, `FREESURFER_HOME`, `MNI_TEMPLATE`, `PY`, `CASES`, `TWINS`, `QUEUE`), read from environment
   variables (`CMB_ROOT`, `CMB_PRIVATE`, ...). Scripts in `code/` and `scripts/paper/` read the same variables directly.
   With the variables set to the original folders the code runs the same commands on the same files.
2. **Manifests and case lists in `dev/`** hold paths relative to the repository root (`dev/gitter_d/sub-XXX`,
   `data/valdo2021/Task2/...`) instead of absolute ones; run from the repository root. `scripts/link_paths.sh` links a
   VALDO copy to `data/valdo2021/Task2`.
3. **In-house cohort names** were removed from code, comments, help texts and figure labels. Where a cohort label is a data
   key (manifest `cohort` field, e.g. `"<label>-t2star"`), it is `CMB_INHOUSE_LABEL` (default `inhouse`); set it to the
   label of your manifests. Run-folder names such as `a03-aniso-e60-cat-t2s-*` are kept, because the stored results are
   found by them.
4. **Twin patients** (scanned in both in-house cohorts; the leave-one-site-out analysis removes them from training):
   the id list was in `cmb/stage2/lodo.py`; it is now read from the private JSON named by `CMB_TWINS`
   (`{"<swi id>": "<t2star id>"}`). Without the file no case is excluded.
5. **Case list of own cohorts** (`faelle.json` in the private folder): `CMB_CASES`. The case-id prefixes of the own
   cohorts on the model grid (`<prefix>-<id>`) are `CMB_ID_PREFIX_T2STAR` / `CMB_ID_PREFIX_SWI` (defaults `t2s`, `swi`; set
   them to the prefixes of existing grids to find stored results).
6. **Local job queue / helpers:** GPU steps that were queued with a local tool (`kern --queue`, `ratio_chain`, the wc5/wc7
   shell scripts) use `CMB_QUEUE`; empty = run directly. Calls of the local blackboard/notification tools were removed.
7. **Removed files** (no number depends on them): `code/fix_block_runde3.py`, `fix_block_runde4.py`, `fix_block_we4.py`
   (edited a lab protocol), `code/turnier_einreihen.sh` (local queue). Status/round-update scripts left out of
   `scripts/paper/` are listed at the end of `scripts/paper/README.md`.
8. **Folder rename:** the in-house preprocessing scripts moved from a folder named after the cohort to `scripts/prep_bids/` (paths there read
   `CMB_SOURCEDATA`, `CMB_BIDS`, `CMB_PRIVATE`, `BIDS_SWI`).
9. **Docs** (`docs/`, written for the thesis phase in September/October): machine paths and cohort names replaced by
   placeholders; aggregate results kept, no per-case in-house results.

## Changes to code behaviour (all additive or bug fixes of the port; none changes a paper number)

| file | change | why |
|---|---|---|
| `code/gitter_c_bauen.py` | `pid[4]` -> `pid[4:5]` (cohort digit in info.json) | ids shorter than 5 characters crashed; identical for VALDO ids |
| `code/frst.py` | `import os` | needed by the path change (found by `cmb selftest`) |
| `cmb/training/full_data.py` | optional `--lr`, `--bs` (default: the recipe) | `cmb train -ilr / -bs` |
| `cmb/experiments/kern.py` | GPU steps run directly when no queue is set | portability |
| new: `cmb/paths.py`, `cmb/cli.py`, `cmb/commands.py`, `cmb/predict.py`, `cmb/pretrained.py`, `cmb/selftest.py`, `cmb/synthetic.py`, `tests/`, `install.sh`, `pyproject.toml` | the `cmb` command line (cmbx/MicrobleedNet option names), installer, tests | usability; wrappers call the paper modules |

## Weights

Released as assets of the GitHub release `paper-v3-2026-11` (not in the git tree), downloaded by `cmb get-model` /
`cmb predict` and checked by SHA-256 (`cmb/pretrained.py`):

| asset | SHA-256 | trained on | provenance check |
|---|---|---|---|
| `valdo_t2_seed42_modell.pt` | `b9301ccb...cfda1` | VALDO 2021 Task 2, all 72 training scans | byte-identical to the paper's Phase 2 start model (`ergebnisse/final/valdo_t2/seed42/modell.pt`); its meta.json lists the 72 VALDO ids and the manifest `dev/manifest_valdo_all72.json` (cohorts valdo-1/2/3 only) |
| `valdo_t2_seed1_modell.pt` | `f0cc5495...c755f` | same, seed 1 | same |
| `valdo_t2_seed*_meta.json` | pinned in `cmb/pretrained.py` | | the original meta.json with the machine path of the manifest made relative |

The tensors equal those of the release `v0.1.0` of microbleed_detection_3D_UNET (`valdo_t2_seed42.pt`, `valdo_t2_seed1.pt`),
but those files use other parameter names and are not byte-identical, so the original files are released here.
The fixed post-processing cell (probability 0.3, >= 2 mm3, <= 2100 voxels) was chosen on the VALDO inner folds.

**Not released: a stage-2 classifier.** The paper's VALDO-only stage-2 classifier (frozen-V, tset-plain, 40 epochs,
3 members, seed offset 0) was retrained in memory for every scoring run (`scripts/paper/phase2_finetune/chain_frozenV.py`)
and never saved as a file. The classifier in microbleed_detection_3D_UNET v0.1.0 was trained on pooled candidates
including the in-house cohorts and must not be used here. `cmb predict` therefore runs stage 1 with the fixed cell (the
paper's n = 0 arm); `cmb cross_validate --stage2` trains stage 2 on your own out-of-fold candidates.

## Found while porting (for the authors)

* **CSF rule in the frozen chain.** Manuscript Section 2.3 says that candidates with a CSF majority label were rejected as
  fixed post-processing. The Phase 2 frozen-V chain built its candidates with `--keep-csf` and scored the classifier
  decisions without the rule (`run_block.py`, `chain_frozenV.py`, `backfill_candidates.py`). The rule was applied in
  the stage-1 fixed-cell scores (`cmb.analysis.operating_point`, `tset_apply` "probability+csf") and in chain B of wk5.
  `cmb predict` follows the frozen chain (no CSF rule) unless `--csf-rule` is given.
* `check_numbers_v3.py` reports 105 of 685 numbers as mismatches against the manuscript of 15.11. (the original script
  gives the same output); the text was edited after the clean run of 11.11.

## Regression check on public VALDO data (CPU, this repository's code)

See the README section "Regression check" for the commands and results.
