# Light post-processing against vessel false positives: what the literature yields (2026-09-29; sonnet subagent, 9 searches; check citations)

| # | Method | Source | Data | Number | transferable to 2-5 mm? |
|---|---|---|---|---|---|
| 1 | 3D radial symmetry, semi-automatic | Kuijf 2013, PLoS ONE | 3 T T2*, 72 pat. | sensitivity 65-84%, 20-96 FP per case (then manual) | unclear |
| 2 | 2D FRST on minimum-intensity projection + region growing + shape features | Bian 2013, NeuroImage Clin 2:282 | SWI (radiation-induced CMB) | sensitivity 86.5% at 44.9 FP per case | plausible (mIP across z) |
| 3 | Threshold + SVM on shape/intensity | Barnes 2011, NeuroImage | 7 T | sensitivity 81.7%, ~107 FP per case | no |
| 4 | msLoG candidates + Radon/Hessian descriptors + cascaded RF | Fazlollahi 2015, CMIG 46:269 | SWI | sensitivity 87%, FP numbers not found | questionable (Hessian discarded in our case) |
| 5 | **Voxel RF (12 features) -> object RF with 7 shape features (elongation versus sphere)** | van den Heuvel 2016, NeuroImage Clin, PMID 27489772 | SWI + T1, SHT | **sensitivity 89.1% at 25.9 FP per case** | medium: object geometry, not curvature |
| 6 | **Bounding-box axis ratio > 3 = vessel, volume, diagonal <= 10 mm, then a small network** | "geometric-to-neural cascade" 2026, PMC13483767 | 1 mm isotropic | cascade overall: FP 1157 -> 269 (-77%), sensitivity 0.93 -> 0.71 | not proven for thick slices |
| 7 | Vein atlas / Frangi mask as exclusion | -- | -- | **no evidence found** | -- |
| 8 | Calcification versus blood via QSM sign | Eur Radiol 2024, 10.1007/s00330-024-10889-z | phase required | no number in the excerpt | only with phase |
| 9 | Mimic locations: basal ganglia/choroid plexus/pineal/falx calcification | mentioned repeatedly | -- | no FP number without QSM | -- |
| 10 | 2D features per slice + number of occupied slices + centroid shift | principle appears in several pipelines | -- | no single number | yes, checkable in minutes |

## The two cheapest, not-yet-measured candidates (from the agent; measured today, `geom_*.json`)
(a) **Axis ratio of the candidate component** (PCA longest/shortest principal axis in mm, threshold around 3; global object geometry, not
    local curvature like Frangi/Hessian), plus the in-plane ratio per slice;
(b) **Shape stability across slices**: number of occupied native slice blocks and the centroid shift between neighbouring slices
    (targets geometry, not dark continuation like the discarded run length).
Caveat up front: the components are those of the U-Net (threshold 0.15), not segmentations of the image -- a vessel that the U-Net only
marks pointwise has a compact component; and the z-extent of the component was larger for REAL haemorrhages in our case (block filling).
