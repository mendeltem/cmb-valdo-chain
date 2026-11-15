# scripts/paper: the scripts that produced the numbers of the clinical CMB paper (manuscript v3)

These are the job scripts as they ran (October/November 2026), one folder per paper phase. File names are kept so that
each one can be traced back to its job (wk0, wk1, wk2, wk0b, wk3, wk5, wk9, wk11a-f; arena jobs wc3-wc7). The logic is
unchanged. The only edits are for portability and privacy (see `PAPER_VERSION.md` in the repository root):

* absolute paths of the original machine -> environment variables (same names as `cmb/paths.py`);
* in-house cohort labels used as data keys -> `CMB_INHOUSE_LABEL` (set it to the label your manifests use);
* the local GPU job queue and job-wait helpers -> optional `CMB_QUEUE` / `CMB_WAIT` (empty = run directly, synchronously);
* names of the in-house cohorts, of the machine and of the agents in comments, help texts and figure labels -> neutral words
  ("in-house", "workstation"); figure labels therefore read "in-house" where the manuscript figures say the clinic's name;
* two in-house case ids in comments removed.

Comments and messages are partly German (the lab language during the study).

## Environment variables

| variable | default | meaning |
|---|---|---|
| `CMB_ROOT` | the repository | public work folder: `dev/` (VALDO grids, manifests, frozen folds), `ergebnisse/` (results), `paper/` (figures, wk11 JSONs) |
| `CMB_PRIVATE` | `$CMB_ROOT/private` | private work folder of your own cohorts: `gitter/` (grids, manifests), `transfer/`, `stage2/`, `paper/` (phase-2/3 results), `extern/` |
| `CMB_CASES` | `$CMB_PRIVATE/faelle.json` | case list of the own cohorts (`{"t2star"|"swi": {"faelle": [...]}}`) |
| `CMB_INHOUSE_LABEL` | `inhouse` | cohort label in the manifests (`<label>-t2star`, `<label>-swi`) |
| `CMB_VALDO` | `$CMB_ROOT/data/valdo2021/Task2` | VALDO 2021 Task 2 |
| `CMB_MOMENI` | `$CMB_ROOT/data/momeni-csiro-cmb` | public Momeni/CSIRO SWI cohort |
| `CMB_PYTHON` | the running interpreter / `python` | interpreter for sub-processes |
| `CMB_EXTERN_MODELS` | `$CMB_ROOT/models/extern` | third-party models for wk1/wk1b (`mednet_cmb/repo`, `mednet_cmb/MedNet-CMB_weights`, `microbleednet_checkpoints`) |
| `CMB_QC_PLOTS` | `$CMB_PRIVATE/qc_plots` | QC images used by `plot_example_subject.py` |
| `CMB_LOGS` | `$CMB_PRIVATE/logs` | logs of background jobs (shell scripts only) |
| `CMB_ROOT_NAME`, `CMB_PRIVATE_NAME` | `root`, `private` | `check_numbers_v3.py` only: first path component of the key paths in NUMBERS_v3.md that means CMB_ROOT / CMB_PRIVATE |
| `CMB_QUEUE`, `CMB_WAIT` | empty | optional GPU queue command (`<cmd> ... -- <command>`) and job-wait command; empty = run directly / no wait |

Run every script from the repository root. Python scripts find `cmb/` and `code/` through `CMB_ROOT`.

## Phase 1 -- arena (stage-1 U-Net and stage-2 classifier selection), `phase1_arena/`

| script | job | what it computes | feeds |
|---|---|---|---|
| `wc3_a_ensembles.sh`, `wc3_c_alt.sh`, `wc3_c2_fortsetzung.sh`, `wc3_c3_fertig.sh`, `wc3_d_neighbours.sh` | wc3 | arena phase 0 (CPU): ensembles, candidate ceilings, alternative candidate sources (frst999/frst995/gmm), pooled classifiers, neighbour features | Table S2/S3 (arena catalogue/results) |
| `wc3b_runde.sh`, `wc3b_mappe.sh`, `wc3b_maps.sh` | wc3b | capped alternative candidate maps (frst/gmm x K) + ceiling analysis | Table S3 |
| `wc3b_probelauf.sh`, `wc3_check.sh` | wc3/wc3b | operational: probe run with timing, job status check (no reported number) | -- |
| `wc5_atlas_ordner_anlegen.sh`, `wc5_register_all.sh`, `wc5_register_t2star_parallel.sh`, `wc5_rest.sh`, `wc5_auc_teilvaldo.sh`, `wc5_regel_entscheid.py`, `wc5_zahlen_sammeln.py` | wc5 | venous-atlas features (`cmb.stage2.atlas_features`): registration, atlas candidate folders, partial AUC, pre-fixed decision rule, numbers per arm | Table S3 (atlas arm) |
| `wc5_probe_register.py` | wc5 | operational: registration probe with timing | -- |
| `wc6_insert_all.sh`, `wc6_candidates_all.sh`, `wc6_realism_s0.sh` | wc6 | synthetic CMB/vein insertion, their candidates, realism gate | Table S3 (synthetic arm) |
| `wc7_candidates_union.sh`, `wc7_folgen.sh`, `wc7_seed1_folgen.sh`, `wc7_summary.sh` | wc7 | sensitive candidate unions (blob + z40), pooled tset runs, summaries (rules v2) | Table S3 (union arm) |
| `wk0_run_all.py` | wk0 | U-Net re-rank of stage-1 runs on all four data sets (`cmb.analysis.unet_rerank`) | 2.5 / Table S3 (stage-1 choice) |
| `auswertung.py`, `pool_all.py`, `verify_thresh.py` | wk1 | external baselines (SHIVA-CMB): nested threshold choice and scoring, pooled over cohorts, threshold verification | 3.2 (public models as downloaded) |
| `mbnet_infer.py`, `mednet_infer.py`, `v2_postprocess.py`, `v2_mbnet.py`, `v2_mednet.py` | wk1b | MicrobleedNet (original weights) and MedNet-CMB zero-shot inference; scoring with exactly the post-processing of our chain | 3.2, Table 2 |
| `phase2_chain.sh`, `phase2_tail.sh`, `phase2_tail2.sh`, `phase2_summarise.py` | wk2 | separate vs pooled ("mixed") stage-2 classifier factorial, three seeds, summary + figure | 3.4 (separate vs mixed classifier) |
| `phase2_ft_chain.sh`, `phase2_ft_summarise.py` | wk2b | pooled pre-training + fine-tuning of the classifier (P->FT arm) | 3.4 |
| `run_seed.sh`, `select_a1v2.py`, `finish_r4.sh` | wk9 | Arena 1 v2: pooled VALDO-only runs per seed, selection of the VALDO-only winner (tset-plain), then the frozen-A1v2 chain + comparison | 2.5 (VALDO-only selection), Fig. S (A1v2) |

## Phase 2 -- fine-tuning learning curves, `phase2_finetune/`

| script | job | what it computes | feeds |
|---|---|---|---|
| `phase4_draws.py` | wk0b | nested scan draws per sequence, frozen fold and n (STRAT/RAND/HIGH/LOW/MID/NONE) | 2.6 |
| `run_block.py` | wk0b | one block = (sequence, fold, variant S / FT / FTn, n, seed): U-Net training from scratch, full fine-tuning of the VALDO-72 U-Net, or norm/output-layer-only fine-tuning; test-fold prediction, z40 candidates, stage-1 score | 3.3, Fig. 3 |
| `queue_driver.py` | wk0b | runs all blocks of one (sequence, fold) in sequence | -- |
| `backfill_candidates.py` | wk0b | refills candidate sets written empty by an early run_block bug (fixed 01.11.) | -- |
| `predict_n0.py`, `calibrate_n0.py` | wk0b | n = 0 maps of the VALDO full-data U-Net and the calibration-only control (threshold only, CAL-NEG) | 3.3, 4.2 |
| `chain_frozenV.py` | wk0b | stage-2 chain frozen-V: the VALDO-only tset-plain classifier applied unchanged to every block | 3.3 (endpoint), Fig. 3 |
| `chain_frozenA1v2.py`, `compare_a1v2_frozenV.py`, `fig_a1v2.py` | wk9 | the same with the Arena 1 v2 winner; reproduction check against frozen-V; figure | 2.5, Fig. S |
| `primary_endpoint.py`, `good_for_us.py`, `wk4b_update_summary.py`, `update_summary.py` | wk0b / wk4b | primary endpoint (plateau n), secondary "good for us" n, phase-4 summary JSON | 3.3, 4.6 |
| `draw_strat_table.py` | wk0b (review) | draw-strategy arms vs STRAT | Table S (draw strategies) |
| `fig5.py`, `fig5_final.py` | wk0b | learning-curve figure | Fig. 3 (learning curve) |
| `gpu_chain_swi_v2.sh` | wk0a | stage 1 + candidates on the SWI grids with hole-filled brain masks | 3.2/3.3 (SWI) |

## Phase 3 -- cohort analysis, `phase3_cohorts/`

| script | job | what it computes | feeds |
|---|---|---|---|
| `wk5_klinisch.py` | wk5 | MARS location, patient-level counts, ICC and agreement | 3.5, Fig. 4 |
| `wk5_check1_dump.py` | wk5 | check: metrics recomputed from the decision dumps equal the stored results | -- (gate) |
| `patient_threshold.py` | wk5b | patient-level operating point for the > 4 CMB rule | 3.5, Table (patient level) |
| `plot_fig4_site_count.py` | wk3 | site-count curves (leave-one-site-out, `cmb.stage2.lodo` / `lodo_tset`) | Fig. S6 / Supplement S2 |
| `check_phase3_model.py` | wk3 | acceptance check: phase-3 summaries come from tset-plain | -- (gate) |

## Manuscript v3 numbers, figures and tables, `manuscript_v3/`

| script | job | what it computes | feeds |
|---|---|---|---|
| `wk11_01_folds.py` | wk11a | all-fold numbers (FT draws pooled per fold) | 3.3 |
| `wk11_02_thresholds.py`, `wk11_02_margins.py` | wk11b | one threshold set for 3.2/3.3/4.2; margins | 3.2, 3.3, 4.2 |
| `wk11_03_sizeclasses.py`, `wk11_03_plateau_check.py` | wk11c | recall by size class (< 2 / 2-5 / > 5 mm, one diameter definition, one-to-one matching); plateau check | 3.3, Table (size classes) |
| `wk11_04_prevalence.py` | wk11d | grey zone and prevalence table | Table S8 |
| `wk11_05_split.py` | wk11d/e | split check | 2.4 |
| `wk11_06_mbnet.py` | wk11e | MicrobleedNet method check (report only) | 3.2 |
| `wk11_07_numbers.py` | wk11f | derived numbers for NUMBERS_v3 | all sections |
| `check_numbers_v3.py` | wk11f | gate: every number of NUMBERS_v3.md equals its result JSON and occurs in the manuscript (needs the private result JSONs and the manuscript; 685 numbers, 0 mismatches on 11.11.2026) | -- (gate) |
| `wk11_probe.py` | wk11 | operational: read-only dump of the source structure | -- |
| `plot_learning_curve.py`, `plot_valdo_base_model.py`, `plot_slice_thickness.py`, `plot_slice_thickness_heatmap.py`, `plot_inhouse_size_heatmap.py`, `plot_wk7_slice_thickness.py`, `plot_example_subject.py` | wk10-wk12 | figures (learning curve, base model, slice thickness, size heatmap, example subjects) | Fig. 2, 3, 5, S3, S5 |
| `render_tables_s.py` | wk10 | Tables S2/S3 rendered as HTML for PNG | Table S2, S3 |

## Left out, and why

* `update_phase3_summary_1107.py` (wk3) -- bookkeeping only: rewrites status texts of the phase-3 summary and a requeue
  entry of the local job system; computes no number.
* `summary_v2.py` (wk1b) -- bookkeeping only: marks old blocks of the external-baseline summary invalid and writes status
  texts keyed by the in-house cohort names; computes no number.
* `pruefe_pfade.py`, `cdisc_diagnose.py`, `test_choose.py`, `mbnet_rerun.py`, `ausgabe_r8.py`, `update_summary_r5/r7/r8.py`
  (wk1/wk1b work folder) -- path checks, diagnostics and round-by-round status updates, not part of the reported chain.
* `code/wk7_*` (slice-thickness simulation) -- already in `code/`.
* Results, logs, decision dumps, draws and summaries (`*.json`, `*.jsonl`, `*.log`) -- results of in-house data, never public.
