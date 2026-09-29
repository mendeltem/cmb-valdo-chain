# Licence for code, weights, and QC pages (2026-09-29; sonnet subagent, 6 searches + 2 pages; NOT legal advice, uncertain points marked)

| Question | Finding | Source |
|---|---|---|
| CC BY-NC-SA 4.0 and trained weights | "Adapted Material" = a modification that requires copyright permission; whether weights fall under this is **not settled** (depends on the jurisdiction). If yes: ShareAlike, so the weights would also be BY-NC-SA. Code is independent of this and may be MIT/Apache. | creativecommons.org FAQ; CC paper "Using CC-licensed Works for AI Training" (2025); wiki.creativecommons.org/wiki/4.0/Treatment_of_adaptations |
| MicrobleedNet (v-sundaresan/microbleed-detection) | no LICENSE file discernible; weights via Google Drive without conditions | github.com/v-sundaresan/microbleed-detection |
| SHIVA-CMB | **code AGPLv3, weights CC BY-NC-SA** -- the direct model in the field | github.com/pboutinaud/SHIVA_CMB, /SHiVAi; Sci Rep 2024 |
| TotalSegmentator | code Apache-2.0; individual models on restrictive data CC BY-NC(-SA) | github.com/wasserth/TotalSegmentator |
| VALDO Task 2 | Zenodo 4520773, CC BY-NC-SA 4.0, citation and acknowledgement obligation (funders, "Where is VALDO", Med Image Anal 2024); cohorts SABRE, Rotterdam, ALFA | zenodo.org/records/4520773; valdo.grand-challenge.org/Task2 |
| Momeni/CSIRO | CSIRO Data Licence: non-commercial, NO redistribution of the raw data; permits "hardcopy products and non-editable digital images (e.g. PDF)" free of charge -- static excerpts tend to be fine, interactive voxel viewer **unclear** | research.csiro.au/dap/licences/csiro-data-licence |
| CenSynCMB | no reliable licence statement found | -- |

## Recommendation for the repo (cautious reading, not a legal statement)
1. Two licence files: `LICENSE` (code) = MIT or Apache-2.0; `LICENSE-WEIGHTS` = CC BY-NC-SA 4.0 with the sentence "Trained on VALDO Task 2
   (CC BY-NC-SA 4.0, non-commercial), zenodo.org/records/4520773; cite Sudre et al. 2024" -- as with SHIVA-CMB. Wording: "chosen as
   BY-NC-SA out of caution", not stated as a settled legal consequence.
2. QC page with VALDO excerpts: non-commercial + source attribution visible -> same logic as the data itself (the repo valdo-cmb-qc already exists).
3. Momeni excerpts: static only (PNG/PDF), with source, no revenue; do NOT publish the slice viewer with Momeni images
   (the voxel data sits in it as a JPEG stack -- that is close to "editable"). The numbers (F1 etc.) are unproblematic.
4. catalina: never, not even as an excerpt (patient data, own rule).
5. Before publication: carry VALDO's citation obligations (funder acknowledgement) over into the README and the paper.
