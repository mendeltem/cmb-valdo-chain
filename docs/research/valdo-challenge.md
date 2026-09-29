# VALDO challenge, task 2 (CMB) -- what the paper says (Claude, 2026-09-18, frugal research)

Source: Sudre et al., "Where is VALDO?", arXiv 2208.07167 (HTML version https://arxiv.org/html/2208.07167),
two targeted retrievals; read out by a small reading model, table values taken verbatim,
NOT checked line by line by me. Evaluation code: https://github.com/WhereIsValdo/valdo-eval-2021
(README without metric details; source code not read).

## Read
- Metric task 2: F1 and absolute difference in count (detection), mean Dice and volume difference.
  Matching: predicted centroid < 5 mm from the reference centroid; components with
  6-connectivity, threshold 0.5. The MEDIAN PER SUBJECT [quartiles] is reported, no pooled F1,
  no sensitivity/precision/FP per scan. Test set: 147 (SABRE 20, RSS 68, ALFA 59), held out.
- Inter-rater agreement for CMB: NOT reported in the paper. Reference: BOMBS (SABRE, ALFA) or
  Vernooij 2008 (RSS); with two annotations, their mean.
- Results (median F1 per subject in %, median Dice): MixMicrobleedNet 68.4 / 84.0 (nnU-Net 3D,
  whole image, all modalities, 1000 epochs); TeamTea 66.7 / 82.6 (3D U-Net, patch 96x192x128,
  bias field correction, all modalities); Zihao 66.7 / 80.0 (TWO-STAGE: 3D-U-Net candidates +
  FCN/AlexNet classifier on 20x20x16 / 24x24x20, wBCE+Dice); ValdoNN 50.0 (nnU-Net 2D);
  Tfff 40.0 (2D); BigrBrain 16.7 (nnU-Net 2D); Dawai 0.0 (3D-U-Net with blob loss); TheGPU 0.0
  (Random Forest, T2* only). Per cohort only for two teams: Zihao ALFA 50 / RSS 74.8 / SABRE 45.
- ALL deep-learning teams used T1 + T2 + T2* (in T2* space). Only the Random Forest team used T2* alone.
- Authors: 3D information "particularly relevant to avoid mimics"; even the best miss cases
  and report CMB in cases without CMB; augmentation with interpolation/deformation can hurt.

## Not found
How F1 is computed for subjects WITHOUT reference CMB; how many test subjects without CMB; details of the
winning method (post-processing, handling of negative cases); exactly what "use of the whole extent of
the training data" refers to.

## Conclusions for us (interpreted)
1. Our 0.25-0.33 (POOLED, 26-connectivity, 57 CV cases) are NOT directly comparable to the 0.67 (median per
   subject, 5 mm centroid, held-out test). First compute the median per case with code/metric_valdo.py,
   then compare.
2. The three best differ from us less in architecture than in: (a) context -- whole image or
   96x192x128 versus our 38x62x94 voxels (47x31x38 mm); (b) training duration -- nnU-Net
   1000 epochs versus at most 40 x 800 patches; (c) inputs -- T1+T2+T2* versus T2*+FRST.
3. Two-stage (Zihao) is on par with the winner -> network 10 is justified.
4. 2D consistently performs worse (16.7 / 40 / 50) -> a04-2p5d will presumably not win.
5. blob loss brought nothing there (Dawai 0.0) -- no reason to prefer it.
