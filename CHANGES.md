# Changes

## After paper-v3-2026-11 (main; the tag stays on the code of the paper)

Found by testing the tag on own data. None of these changes touches the paper code that produced the results
(`cmb/` research modules, `code/`, `scripts/paper/`); they fix the `cmb` command line added for the release.

1. **`cmb evaluate` now scores every labelled scan, CMB-free scans included**, so their false positives count, as in the
   paper (`cmb.transfer.evaluate`: `code/metric.hits` per scan, pooled with `summarize`, all scans of the manifest). Before,
   scans without reference CMB were left out of the F1. Scans prepared without `-l` (manifest `"labelled": false`) are
   not scored. `summary.json` also gives `n_scored` and `n_cmb_free_scored`. Test: `tests/test_evaluate.py`.
2. **`cmb preprocess` searches recursively** (flat folders and BIDS trees; `derivatives/`, `sourcedata/` and `code/` are
   not searched for raw scans), and finds brain masks (`-b`) and labels (`-l`) anywhere below their folders.
3. **Scan matching on the suffix at the END of the file name only** (`-ir _T2` matches `x_T2.nii.gz`, not `x_T2_roi.nii.gz`
   or `x_T2_brainmask.nii.gz`); names ending in the `-lr` / `-br` suffix are never scans; the case id is the name
   without the suffix (no splitting of the path). `-br` / `-lr` may be given with or without `.nii.gz`. Test:
   `tests/test_preprocess.py`.
4. **Grid profiles.** The release's `cmb preprocess` / `cmb predict` built the VALDO grid (`dev/gitter_d`,
   `code/preproc_roh_bauen.py`: no reorientation, brain = image > 0 with holes filled, `1 - I/max`). The paper built the
   grids of its own cohorts differently (`cmb.transfer.build_private_grids.prepare_source`):
   * the bias-corrected image, the brain mask and the label are reoriented to RAS+ (`nibabel.as_closest_canonical`) --
     this is the mirrored x axis seen on scans stored in LAS/LPS;
   * brain = mask AND image > 0 (no hole filling), `1 - I / max(I in the brain)`, voxels at 0 inside the brain set to
     1e-6, saved without the scan header -- these are the value differences.
   Bias correction is the same in both (FSL FAST -B on the brain-extracted image). `--profile own` (now the default of
   `cmb preprocess` and `cmb predict`) reproduces the own-cohort pipeline bit for bit (test on a LAS phantom against
   `prepare_source` + `code/gitter_c_bauen.py`); `--profile valdo` reproduces the stored VALDO grid bit for bit
   (test on the public VALDO raw case sub-101 against the study's `dev/gitter_d`, run when `CMB_REFERENCE_GRID` points
   at it). Outputs of `cmb predict` are written back on the scan's own grid in both profiles.
6. **VALDO-only stage-2 classifier (release v1.1.0).** `cmb get-model valdo_stage2` downloads a stage-2 candidate
   classifier trained on VALDO alone (`cmb.stage2.final_classifier` on the 278 out-of-fold candidates of the 57 VALDO
   cross-validation cases, 3 members, 40 epochs, threshold 0.55 by 5-fold CV by case, pooled out-of-fold lesion F1 0.614;
   SHA-256 `ef179dca588ab522a27344c94f6001a7c01a4c701b27655e3808216302d6d4ef`, CC BY-NC-SA 4.0). It is NOT the paper's
   frozen-V classifier (never saved). `cmb predict --stage2` (off by default) replaces the fixed cell with the classifier's
   decision on the candidates (p > 0.15, >= 1 mm3, built exactly as `cmb.stage2.candidates`; test
   `tests/test_stage2.py`); `--stage2 FILE` takes any `classifier.pt` of `cmb.stage2.final_classifier`. The weights now
   download from release v1.1.0 (the paper tag keeps the stage-1 files).
5. `scripts/prep_bids/hirn_und_bias.py`: HD-BET and FSL are found through `HDBET` / `PATH` and `FSLDIR` (it had
   home-relative paths of the original machine); `tests/test_privacy.py` now also rejects home-relative paths.
7. **Code-quality checks (main, after v1.1.0).** No output changes on valid input; the reference tests (own profile
   against `prepare_source`, VALDO profile against the stored grid, stage-2 candidates against `cmb.stage2.candidates`)
   still pass bit for bit.
   * Run-stopping input checks (`cmb/checks.py`, test `tests/test_checks.py`): `cmb preprocess` and `cmb predict` stop
     when a brain mask, label or FRST map is not on the scan's voxel grid (matrix, orientation code, corners within a
     quarter voxel: the rule of `scripts/prep_bids/eingaben_bauen.py`), when a brain mask is empty or mostly on zero
     scan voxels, or when a labelled CMB lies entirely outside the scan's data; the manifest commands stop on an empty
     or repeated case id; `cmb evaluate` stops when a scan labelled in the manifest has no label file (before, it was
     left out of the F1 silently) and checks that every labelled scan was scored. A grid folder without manifest.json
     marks scans without a label file as unlabelled, as they were treated before.
   * Names ending in `_roi` are never scans (with `_brainmask` and `_mask`), for every `-ir`.
   * The Python API stops with a message when `preprocess` / `fine_tune` get no `o=` or `train` no `m=` (it passed the
     text "None" as the folder name).
   * Static analysis (ruff, pyright basic, import-linter contracts in `pyproject.toml`), property tests
     (`tests/test_properties.py`, hypothesis), the freeze check of the research code (`tests/test_frozen.py`), and CI
     (`.github/workflows/checks.yml`: checks + `pytest tests -m "not valdo"`; VALDO tests carry the marker `valdo`).
8. **ICC(3,1) of the paper scripts was wrong (16.11.2026; an agreed change of frozen research code).** The own `icc31`
   copies in `scripts/paper/phase2_finetune/good_for_us.py`, `phase2_finetune/wk4b_update_summary.py` and
   `phase3_cohorts/wk5_klinisch.py` multiplied the between-patient sum of squares by n (patients) instead of k = 2
   (raters), did not divide it by its degrees of freedom n - 1, and used n - 1 instead of k - 1 in the denominator; they
   returned about 0.997 for any data. `scripts/paper/phase1_arena/auswertung.py` had another wrong pairs formula. All
   four now import one implementation, `cmb/agreement.py` (Shrout & Fleiss 1979 ICC(3,1) = McGraw & Wong 1996
   ICC(C,1): two-way mixed, consistency, single rater). Test `tests/test_agreement.py`: Shrout & Fleiss Table 2
   (ICC(3,1) = .71), an independent two-way ANOVA (statsmodels) and pingouin when installed, and independent counts
   that must not come out near 1. Corrected per-patient count ICC(3,1) of the chain: T2*GRE 0.815, SWI 0.834, VALDO
   0.823 (was 0.997 each). In the "good for us" rule (criterion 3: ICC >= 0.75) a few small-n cells now fail
   criterion 3, but none of the reported smallest-n decisions changes, since criterion 1 or 2 already failed there.
   `tests/frozen.sha256` regenerated for these four files; the tag `paper-v3-2026-11` keeps the original code.
