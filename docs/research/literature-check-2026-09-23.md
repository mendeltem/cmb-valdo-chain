# Literature check 23 Sep -- cross-check of the [R]/[G] entries

Purpose: sample check of the 20 most important unchecked entries from `LITERATURE.md` (status [R] or [G])
against real sources online. Selection criterion: tier A or tier B from section 0 of the literature list,
or evidence for a method that was actually rebuilt (loss, preprocessing, second stage). Pure background
literature (section 3 without a tier A/B mention, tier C collective items) was not checked.

**Status: completed.** All 20 selected entries checked, 23 of 25 web retrievals used (budget not exhausted,
because all 20 were clean and could be resolved without follow-up queries).

Finding values: CONFIRMED / CORRECTED / NOT FOUND / NOT CHECKED (budget)

## Results table

| Entry (short form) | previous status | finding | what needs to change |
|---|---|---|---|
| Isensee et al., nnU-Net, Nature Methods 2021 | G | CONFIRMED | Nothing. Nature Methods 18:203-211 (2021), DOI 10.1038/s41592-020-01008-z -- matches LITERATURE.md exactly. |
| Greenberg et al., Cerebral microbleeds guide, Lancet Neurol 2009 | G | CONFIRMED | Nothing. Lancet Neurol 8(2):165-174 (2009) matches exactly. PMID 19161908 can be added as a reference (LITERATURE.md currently has no link). |
| Wardlaw et al., STRIVE, Lancet Neurol 2013 (+ Duering STRIVE-2 2023) | G | CONFIRMED | Nothing. Wardlaw: Lancet Neurol 12(8):822-838 (2013) matches exactly. Duering STRIVE-2: Lancet Neurol 22(7):602-618 (2023), PMID 37236211 -- volume/pages were still open in LITERATURE.md, can be added. |
| Varoquaux, Cross-validation failure, NeuroImage 2018 | G | CONFIRMED | Nothing. NeuroImage 180:68-77 (2018), DOI 10.1016/j.neuroimage.2017.06.061 -- matches exactly. |
| Kofler et al., blob loss | R | CONFIRMED | Open conference question from LITERATURE.md now resolved: the conference is IPMI 2023 (Information Processing in Medical Imaging), Springer LNCS 13939, chapter DOI 10.1007/978-3-031-34048-2_58; preprint arXiv:2205.08209 (2022). The note "(check conference before citing)" can be replaced with "IPMI 2023". |
| Dou et al., 3D CNN CMB detection, IEEE TMI 2016 | R | CONFIRMED | Nothing. IEEE TMI 35(5):1182-1195 (2016), PMID 26886975 -- matches exactly; pages were not given in LITERATURE.md, can be added. |
| Al-masni et al., two-stage YOLO + 3D-CNN, 2020 | R | CONFIRMED | Nothing. "A Two Cascaded Network Integrating Regional-based YOLO and 3D-CNN for Cerebral Microbleeds Detection", NeuroImage: Clinical 2020, PMID 33018167 -- matches the sources already linked in LITERATURE.md. |
| Liu et al., SWI + Deep Learning CMB, NeuroImage 2019 | R | CONFIRMED | Nothing. NeuroImage 198:271-282 (2019), DOI 10.1016/j.neuroimage.2019.05.046 -- matches exactly. |
| Setio et al., lung nodule FP reduction, IEEE TMI 2016 | R | CONFIRMED | Nothing. IEEE TMI 35:1160-1169 (2016), PMID 26955024 -- matches exactly. |
| Ghafoorian et al., location-aware 3D CNN lacunes, NeuroImage Clinical 2017 | R | CONFIRMED | Nothing. NeuroImage: Clinical 14:391-399 (2017), PMID 28271039 -- matches; volume/pages were not given, can be added. |
| Billot et al., SynthSeg, Medical Image Analysis 2023 (+ robust variant PNAS 2023) | G | CONFIRMED | Nothing. SynthSeg: Medical Image Analysis 86:102789 (2023), PMID 36857946. Robust variant: "Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets", PNAS 2023, DOI 10.1073/pnas.2216399120, PMID 36802420 -- both title/year confirmed. |
| Loy & Zelinsky, FRST, IEEE TPAMI 2003 | G | CONFIRMED | Nothing. IEEE TPAMI 25:959-973 (2003), DOI 10.1109/TPAMI.2003.1217601 -- matches exactly. |
| Lakshminarayanan et al., Deep Ensembles, NeurIPS 2017 | G | CONFIRMED | Nothing substantively wrong, "Deep Ensembles" is a short title. Full title to add: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", NeurIPS 30 (2017), pp. 6402-6413. |
| Gregoire et al., MARS, Neurology 2009 | G | CONFIRMED | Nothing. Neurology 73(21):1759-1766 (2009), DOI 10.1212/WNL.0b013e3181c34a7d -- matches exactly. |
| Haller et al., Cerebral Microbleeds Imaging, Radiology 2018 | G | CONFIRMED | Nothing. Radiology 287(1):11-28 (2018), DOI 10.1148/radiol.2018170803 -- matches exactly. |
| Momeni et al., synthetic microbleeds, Comput Methods Programs Biomed 2021 | R | CONFIRMED | Nothing. Comput Methods Programs Biomed 207:106127 (2021), PMID 34051412 -- matches; volume/article number were not given, can be added. |
| Milletari et al., V-Net, 2016 | G | CONFIRMED | Nothing. Proceedings: Proc. IEEE Int. Conf. on 3D Vision (3DV) 2016, authors Milletari/Navab/Ahmadi -- venue was not given in LITERATURE.md (only the year), can be added. |
| Isensee et al., HD-BET, Hum Brain Mapp 2019 | G | CONFIRMED | Nothing. Human Brain Mapping 40(17):4952-4964 (2019), DOI 10.1002/hbm.24750 -- matches exactly. The actual paper title: "Automated brain extraction of multi-sequence MRI using artificial neural networks"; "HD-BET" is the tool name used in LITERATURE.md, not an error. |
| Zhang/Brady/Smith, FSL FAST, IEEE TMI 2001 (+ Jenkinson FSL, NeuroImage 2012) | G | CONFIRMED | Nothing. Zhang/Brady/Smith: IEEE TMI 20(1):45-57 (2001) matches exactly. Jenkinson et al., "FSL": NeuroImage 62(2):782-790 (2012), DOI 10.1016/j.neuroimage.2011.09.015 -- volume/pages were not given, can be added. |
| Breiman, Random Forests, Machine Learning 2001 (+ Friedman, Gradient Boosting, Annals of Statistics 2001) | G | CONFIRMED | Nothing. Breiman: Machine Learning 45(1):5-32 (2001), DOI 10.1023/A:1010933404324. Friedman ("Greedy Function Approximation: A Gradient Boosting Machine"): Annals of Statistics 29(5):1189-1232 (2001), DOI 10.1214/aos/1013203451 -- both confirmed, volume/pages were not given, can be added. |

## Summary

All 20 checked entries are CONFIRMED -- no correction, no "not found". For one entry (Kofler, blob loss) the
conference was still open according to LITERATURE.md ("check before citing"); that is now resolved (IPMI 2023). For
several entries (Dou, Al-masni, Ghafoorian, Momeni, Milletari, Jenkinson/FSL, Breiman, Friedman), LITERATURE.md was so
far missing volume or page numbers, which can be added here without anything having been wrong.

## Not part of this check

The selection was limited to the 20 most important entries (tier A/B or rebuilt methods). Not checked, because of
lower priority per section 0 (tier C, pure background/idea sources) or not individually named in tier A/B: the nine
tournament architectures (Attention U-Net, UNet++, SegResNet, Swin UNETR, ResNet/SE, InstanceNorm/AdamW/SGDR,
Ronneberger U-Net, Cicek 3D U-Net), FreeSurfer, Frangi filter, Hinton distillation, Sundaresan Eur Radiol Exp 2025,
as well as the pure idea sources from section 3/6/7/8 without a tier A/B mention (among others Rashid/DEEPMIR, Myung,
VALDO teams, CenterNet/Kendall, OHEM/Focal Loss, Models Genesis/Med3D, nnDetection/Retina U-Net, Wang-TTA,
Domain-Shift-Cardiac-MRI, Guan/Liu survey, Efron/Tibshirani bootstrap). That is roughly 27 of the 47 unchecked
entries -- candidates for a possible second round.

## Web retrieval counter

23 of 25 used (20 searches for the 20 main entries, of which 3 additional searches for co-listed secondary sources in
bundled rows: Duering/STRIVE-2, Jenkinson/FSL, Friedman/Gradient Boosting).
