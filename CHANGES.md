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
5. `scripts/prep_bids/hirn_und_bias.py`: HD-BET and FSL are found through `HDBET` / `PATH` and `FSLDIR` (it had
   home-relative paths of the original machine); `tests/test_privacy.py` now also rejects home-relative paths.
