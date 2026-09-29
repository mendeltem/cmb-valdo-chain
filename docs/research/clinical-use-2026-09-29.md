# What the number of microbleeds is clinically needed for -- and what that means for our chain (2026-09-29; sonnet subagent, 8 searches; check citations before use)

| # | Clinical use | Threshold | Source | For the chain (sensitivity 0.65 / precision 0.65 / case accuracy 0.7 at the F1 optimum) |
|---|---|---|---|---|
| 1 | Boston criteria 2.0, CAA | >= 2 strictly lobar haemorrhagic lesions (sensitivity 64.5%, specificity 95%) | Charidimou 2022, Lancet Neurol, PMID 35841910 | binary 0 / 1 / >= 2 -- a single false positive pushes a case over the threshold |
| 2 | Anticoagulation in AF (CROMIS-2) | burden classes 0 / 1-4 / >= 5; ICH rate 9.8 versus 2.6 per 1000 patient-years, HR 3.67 | Wilson/Charidimou 2018, Lancet Neurol, PMC5956310 | coarse classes, not the exact number; case accuracy 0.7 is not enough on its own |
| 3 | Thrombolysis (ESO 2021) | > 10 CMB: recommendation against i.v. lysis (weak, low evidence) | ESO 2021, PMC7995316 | overcounting can deny therapy |
| 4 | Anti-amyloid antibodies (ARIA-H) | > 4 CMB = exclusion (lecanemab/donanemab trials, appropriate-use) | FDA label Leqembi 2023; ARIA overview PMC12660453 | sharpest threshold: 4 versus 5 |
| 5 | CAD target values (pre-screening) | > 90% sensitivity at < 5 FP per case is considered acceptable; 3D CNN 95.8% / 1.6 FP with phase | CAD overview; Liu 2019 | our F1 operating point (0.65) is NOT a pre-screening operating point |
| 6 | Burden class per patient instead of lesion count | 2D T2* model: lesion F1 0.70; class >= 4: sensitivity 0.93 / specificity 0.94 | Front. Aging Neurosci. 2026, 10.3389/fnagi.2026.1729422 | our output should be measured this way too |
| 7 | Prevalence | Rotterdam 18.7%, median 1 CMB; stroke cohort 49% | PMC3881231, PMC13239722 | at a median of 1, a single false positive flips the class |
| 8 | explicit "tolerable counting error" | not defined; acceptance via sensitivity/FP targets or class boundaries | -- | still report counting error per case |

## Conclusions
1. **"Sensitivity before precision" is correct for deployment** (pre-screening, rater discards false positives; missed ones are lost) -- but the
   benchmark for that is > 90% sensitivity, not 0.65. Our F1 optimum is the number for the paper, not the operating point for the clinic.
   We have already measured the pre-screening operating point, just not called it that: candidate threshold 0.15 reaches 103 of 139 VALDO haemorrhages
   (0.74) at ~4.9 candidates per case (5w); union of three seeds 120 of 139 (0.86). Report as its own row, "pre-screening mode".
2. **Burden class per patient as a clinical metric** (0 / 1-4 / >= 5 and the ARIA threshold > 4): compute sensitivity/specificity of the class from our
   existing predictions (per-case numbers live in `evaluate`). This is the number a clinician understands -- and the Frontiers 2026 paper
   shows that a lesion F1 around 0.70 can yield 0.93 / 0.94 there.
3. No autonomous decision: all four threshold decisions found require manual recounting; state this in the paper (Discussion, limitations).
