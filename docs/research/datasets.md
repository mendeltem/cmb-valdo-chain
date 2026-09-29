# Further public CMB datasets, models, and code (Claude, 2026-09-19, frugal research: 4 searches, 3 retrievals)

Read = page retrieved itself (evaluated by a small reading model). Hit = only seen in search results,
NOT checked. Nothing downloaded.

## Datasets
| Dataset | Content | Access | Status |
|---|---|---|---|
| AIBL "Synthetic Cerebral Microbleed on SWI images" (Momeni et al., CSIRO) | 75 SWI with 175 real CMB (locations by experts), plus 10 synthetic CMB per scan (825) | open, CSIRO Data Licence, DOI 10.25919/AEGY-NY12, https://data.csiro.au/collection/csiro:50304 ; paper doi 10.1016/J.CMPB.2021.106127 | read; resolution, file format, and whether masks or only coordinates: not stated |
| Dou et al. 2016, "cmb-3dcnn" (CUHK) | 20 SWI (3T Philips) from a study of 320; CMB as coordinates, about 3.7 CMB per scan | http://www.cse.cuhk.edu.hk/~qdou/cmb-3dcnn/cmb-3dcnn.html | hit (cited this way in two papers); page not retrieved. Identical to "SHK" in Sundaresan |
| VALDO Task 2 | 72 T2*-GRE with masks | we have it | -- |
| "Large public dataset of annotated clinical MRIs ... acute stroke" (Sci Data 2023) | 2106 scans with SWI, but stroke annotations, NO CMB masks | open | hit; usable only as unannotated SWI (pretraining, synthetic CMB) |
| TICH2, UK Biobank, OXVASC, Rotterdam, MESA ... | large, with CMB | only on application | not free |

## Pretrained models and code
| What | Content | Access | Status |
|---|---|---|---|
| SHIVA-CMB (Tsuchida et al., Sci Rep 2024) | 3D U-Net, trained on 450 scans from 7 acquisitions / 6 cohorts, T2*-GRE AND SWI; test (96 scans) sens 0.67, prec 0.82, F1 0.74, < 1 FP per image; input 160x214x176 | code https://github.com/pboutinaud/SHIVA_CMB , weights via cloud link in the repo; pipeline https://github.com/pboutinaud/SHiVAi | hit (numbers from the search summary, licence not checked) |
| Zihao Chen, VALDO team "Zihao" | multi-stage: candidates + classifier (challenge rank 2-3, median F1 0.667) | https://github.com/ZihaoChen0319/CMB-Segmentation | hit |
| MixMicrobleed (VALDO team) | Mask-RCNN + U-Net | arXiv 2108.02482 | hit |
| Yonsei-MILab Cerebral-Microbleeds-Detection (Al-masni) | YOLO + 3D-CNN | https://github.com/Yonsei-MILab/Cerebral-Microbleeds-Detection | hit |

## New method papers matching our open points
- "A geometric-to-neural cascade ..." (2026, PMC13483767; read): unsupervised candidates (GMM threshold + anatomical
  mask), then two lightweight 3D-ResNets; 10 in-house + 20 Dou cases; sens 0.71, prec 0.68, F1 0.69; the classifiers
  remove 76.8% of false positives. Supports our planned network 10 (second stage).
- "CenSynCMB: Centre Maps and Physics-Guided Synthesis ..." (arXiv 2607.05325; hit): centre maps instead of masks AND
  physically grounded synthetic CMB -- exactly our ideas 4 and 5; read before building.
- "Learning to Look Closer: A New Instance-Wise Loss for Small Cerebral Lesion Segmentation" (arXiv 2511.17146; hit).
- ESOC 2026 abstract A809: 3D attention U-Net on SWI with MARS regions (hit).

## Assessment
Freely and immediately usable are only AIBL (75 SWI / 175 CMB) and Dou (20 SWI) -- both SWI, both presumably point
rather than mask annotation. For a T2* model they are suitable as additional pretraining and as an external test
cohort, not as a replacement. The bigger lever is SHIVA-CMB: a freely available model trained on 450 scans of both
sequences -- as a benchmark on VALDO and catalina (how good is "off the shelf"?) and as a possible starting point for
fine-tuning.

## Addendum 19 Sep evening: SHIVA-CMB measured
Licence read (CC BY-NC-SA 4.0), weights v2 loaded (SHA-256 matches). On our 57 VALDO CV cases F1 0.575 /
median 0.667 -- BUT according to the paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC11680838/) SHIVA was trained on
60 of the 72 public VALDO cases (plus DOU 17, AIBL 44 real + 60 synthetic, BBS 269). Only catalina is fair.
The same paper also shows: the free datasets DOU and AIBL are usable as training data (used that way there).
