# How well do raters agree on microbleeds? Upper bound for 0.65 (2026-09-29; sonnet subagent, 8 searches; check citations before use)

| # | Finding | Number | Source |
|---|---|---|---|
| 1 | MARS, inter-rater "CMB present" | kappa 0.68 [0.58-0.78] at TE 40 ms, up to 0.87 at TE 26 ms (definite CMB) | Gregoire 2009, Neurology 73:1759 |
| 2 | MARS, count | ICC intra 0.98, inter 0.93 | same |
| 3 | BOMBS, ">= 1 CMB" | kappa 0.68 [0.49-0.86] any, 0.78 lobar | Cordonnier 2009, JNNP, PMID 19008468 |
| 4 | Thick-slice GRE versus thin-slice SWI | only 33% (103/310) of SWI CMB visible on GRE | Nandigam 2009, AJNR 30:338 |
| 5 | T2*-GRE versus SWI, n = 246 | 365 versus 699 CMB; CAA cases 1146 versus 1432 | Shams 2015, AJNR 36:1089 |
| 6 | best SWI model versus most experienced rater | sensitivity 95.8%, precision 70.9%, 1.6 FP per case (with phase) | Liu 2019, NeuroImage: Clinical |
| 7 | Proportion of "possible" CMB | 16.7% of candidates disputed; 53.8% of those gone on follow-up | Neuroradiology 2021, PMC9117345 |
| 8 | VALDO organisers | reference uncertain, strong inter-/intra-rater variability, no reliable ground truth | Sudre 2024, Med Image Anal 91:103029 |
Not found: a human-versus-human F1 or sensitivity number at the lesion level (Dou 2016, Momeni, CenSynCMB mention it only qualitatively).

## For us
- 0.65 lies within the band of documented human-human agreement (kappa 0.68-0.87), even though kappa and F1 do not measure the same thing.
  The upper bound is not a fixed value: sequence and slice thickness change the found number by a factor of 2-3, a sixth of the candidates are
  disputed by definition. Plausible reference band: 0.65-0.85. **Our number sits at the lower edge of this band, not below it.**
- A firmer statement needs the second annotation on our own data -- that is exactly the blinded review page from 22 Sep (215
  views, with the user). It is therefore not just a check, but the number the literature lacks.
- Into the paper (Discussion): Nandigam/Shams as evidence that the cohort shift (T2* 5 mm versus SWI 2 mm) is also a reference-standard
  question; point 7 as a reason to mark "possible" CMB separately in our own cohort.
