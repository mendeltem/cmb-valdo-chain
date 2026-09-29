# Preparing an in-house cohort (never public)
1. BIDS: `~/data/bids/catalina-<mod>/sub-XXX/anat/sub-XXX_T2starw.nii.gz` (`_acq-swi_T2starw` for SWI), `participants.tsv`.
2. Derivatives once: `python catalina/hirn_und_bias.py <t2star|swi>` (HD-BET where the scanner gives no mask; FSL FAST -B). Existing derivatives are skipped.
3. Reference masks: `derivatives/manual-cmb/sub-XXX/anat/…CMB.nii.gz`.
4. Case list `<MB>/faelle.json`: `{"t2star": {"faelle": [{"id","fold","n_cmb","bild","maske","hirn"}]}, "swi": {...}}` — folds stratified by CMB load (see `catalina/kohorte_bauen.py`).
5. Grids: `python -m cmb.transfer.build_private_grids --modality t2star` (and `swi`) → `<MB>/gitter/manifest_<mod>.json`.
6. SynthSeg --robust on the bias-corrected native image: `bash catalina/synthseg_swi.sh` (adapt paths) → `<MB>/synthseg-<mod>/robust/`.
7. Train, candidates, pooled classifier: see README "Reproduce (in-house cohort)". Only sums leave the machine.
