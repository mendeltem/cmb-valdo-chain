# Literatur zur CMB-Linie -- was wir benutzt, nachgebaut oder geprueft haben (Stand 21.09.2026)

Zweck: Leseliste fuer den Nutzer und Grundlage des Literaturverzeichnisses. Jede Zeile sagt, WOFUER die Quelle bei uns steht
(Abschnitt in `ARBEIT.md`) und wie sicher die Angabe ist:

- **[Q]** an der Quelle geprueft -- Seite selbst abgerufen (Text teils von einem kleinen Lesemodell ausgezogen) oder Code selbst gelesen
- **[R]** aus den Recherche-Notizen (`recherche/*.md`): meist nur Suchausschnitt oder Abstract -- vor dem Zitieren lesen
- **[G]** Standardquelle aus dem Gedaechtnis -- Autoren, Jahr, Band und Seiten vor dem Zitieren pruefen

Keine Zahl aus [R]- oder [G]-Quellen ist von uns nachgeprueft, sofern nicht anders vermerkt.

**Gegenprobe 23.09.2026** (`recherche/literatur-pruefung-0923.md`): die 20 wichtigsten Eintraege mit Status [R] oder [G] --
alle aus Stufe A oder B, plus die Belege fuer tatsaechlich nachgebaute Methoden -- wurden an den echten Quellen geprueft.
**Alle 20 bestaetigt**, keine Korrektur, keine nicht auffindbare Quelle. Zwei Ergebnisse daraus: die offene Tagungsfrage zu
Kofler "blob loss" ist geklaert (IPMI 2023, Springer LNCS), und bei acht Eintraegen liegen jetzt Band- und Seitenangaben vor,
die hier fehlten (nichts davon war falsch). Die geprueften Eintraege tragen unten **[Q 23.09.]**. Rund 27 Eintraege bleiben
ungeprueft -- vor allem die neun Turnier-Architekturen der Stufe C und reine Ideenquellen; sie stehen in der Pruefdatei als
Kandidaten fuer eine zweite Runde.

## 0 Nicht alle gleich wichtig -- Reihenfolge zum Lesen

**Stufe A -- zuerst lesen (tragen die Argumentation der Arbeit; ein Gutachter wird danach fragen):**
1. Sudre et al., "Where is VALDO?" -- Datensatz, offizielle Metrik, was die Challenge-Teams erreichten. Ohne das ist keine unserer Zahlen einzuordnen.
2. Sundaresan et al. 2023 (MicrobleedNet) -- das Verfahren, das wir unter die Lupe nehmen; vor allem Methoden und die von den Autoren selbst genannten Grenzen.
3. He et al. 2026 (CenSynCMB) -- aktueller Stand auf DENSELBEN Daten (74.3), drei Ideen davon haben wir nachgebaut; zeigt, wo unser Abstand liegt (Precision, drei Sequenzen).
4. Tsuchida et al. 2024 (SHIVA-CMB) -- wie ein Modell ueber viele Kohorten und beide Sequenzen trainiert wird; unsere fremde Messlatte.
5. Isensee et al. 2021 (nnU-Net) -- Standard der Segmentierung; Begruendung fuer anisotrope Kerne, Normierung, Rezept. Wird als Vergleich erwartet.
6. Greenberg et al. 2009 und Wardlaw et al. 2013 (STRIVE) -- was eine Mikroblutung ist und womit sie verwechselt wird; Grundlage fuer die Diskussion der Fehlalarme.
7. Varoquaux 2018 -- warum Kreuzvalidierung mit ~60 Faellen grosse Fehlerbalken hat; stuetzt unsere Vorsicht bei Unterschieden unter 0.03.

**Stufe B -- fuer die Tiefe einzelner Kapitel:**
8. Kofler et al. (blob loss) -- der laesionsweise Verlust (5j). 9. Dou et al. 2016 -- klassische zweistufige CMB-Kaskade, harte Negative (5d, 5l).
10. "A geometric-to-neural cascade" 2026 und 11. Al-masni et al. 2020 / Liu et al. 2019 -- weitere zweistufige CMB-Systeme zum Vergleich unserer zweiten Stufe.
12. Setio et al. 2016 und 13. Ghafoorian et al. 2017 -- Fehlalarm-Reduktion auf zentrierten Kandidaten, Ort als Eingabe (unser Gewebemerkmal).
14. Billot et al. (SynthSeg, robuste Variante) -- worauf Liquor-Regel und Gewebemerkmal beruhen, und wo die Karte auf T2* / SWI irrt.
15. Loy & Zelinsky 2003 (FRST) -- der zweite Eingabekanal. 16. Lakshminarayanan et al. 2017 (Deep Ensembles) -- warum das Mittel aus Startwerten stabiler ist.
17. Gregoire et al. 2009 (MARS) und Haller et al. 2018 -- Ortseinteilung und klinische Bedeutung. 18. Momeni et al. 2021 -- kuenstliche Blutungen (fuer den Ausblick).

**Stufe C -- nur nachschlagen bzw. als Zitat fuer einen Baustein:** U-Net, 3D U-Net, Attention U-Net, UNet++, SegResNet, Swin UNETR, ResNet, Squeeze-Excitation (Turnier-Architekturen); InstanceNorm, AdamW, Kosinus-Plan,
V-Net / Dice, Focal Tversky, Focal Loss, OHEM, CenterNet, Unsicherheitsgewichtung (Verluste und Training); HD-BET, FSL FAST, Frangi, FreeSurfer (Werkzeuge); Random Forest, Gradient Boosting, Models Genesis, Med3D,
nnDetection, Retina U-Net, Testzeit-Augmentierung (zweite Stufe und Inferenz); Bootstrap, Domain-Adaptation-Survey, Bouthillier et al. (Statistik); dazu alle mit [R] markierten, von uns nicht gelesenen Treffer der Recherche.

## 1 Daten, Challenge, klinische Definitionen

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| Sudre CH et al., "Where is VALDO? VAscular Lesions Detection and segmentatiOn challenge at MICCAI 2021", Medical Image Analysis 2024 | https://arxiv.org/abs/2208.07167 ; Auswertecode https://github.com/WhereIsValdo/valdo-eval-2021 | Datensatz Task 2, offizielle Metrik (`code/metric_valdo.py`), Rangvergleich (2a, 5) | Q |
| VALDO-Daten, Lizenz CC BY-NC-SA 4.0 | Zenodo 10.5281/zenodo.4520773 ; `dev/lizenz-valdo.md` | Nutzungsbedingungen, Pflichtnennungen | Q |
| Momeni S et al., "Synthetic microbleeds generation for classifier training without ground truth", Comput Methods Programs Biomed 2021; Datensatz AIBL-SWI mit synthetischen CMB | https://www.sciencedirect.com/science/article/abs/pii/S0169260721002029 ; Daten DOI 10.25919/AEGY-NY12 | freier SWI-Datensatz, Idee "kuenstliche Blutungen" (`recherche/datensaetze.md`) | R |
| Dou Q et al., "Automatic Detection of Cerebral Microbleeds From MR Images via 3D Convolutional Neural Networks", IEEE TMI 35(5), 2016 | https://pubmed.ncbi.nlm.nih.gov/26886975/ ; Daten http://www.cse.cuhk.edu.hk/~qdou/cmb-3dcnn/cmb-3dcnn.html | klassische zweistufige Kaskade mit harten Negativen (5d, 5l); freier SWI-Datensatz | R |
| Greenberg SM et al., "Cerebral microbleeds: a guide to detection and interpretation", Lancet Neurol 8:165-174, 2009 | -- | Definition und Verwechsler der CMB | G |
| Wardlaw JM et al., STRIVE, Lancet Neurol 12:822-838, 2013; Duering M et al., STRIVE-2, Lancet Neurol 2023 | -- | Groessendefinition 2-5 (bis 10) mm (5i) | G |
| Gregoire SM et al., "The Microbleed Anatomical Rating Scale (MARS)", Neurology 73:1759-1766, 2009 | -- | Ortseinteilung lobaer / tief / infratentoriell (`code/gewebekarte.py`, 5i) | G |
| Haller S et al., "Cerebral Microbleeds: Imaging and Clinical Significance", Radiology 287:11-28, 2018 | -- | Uebersicht, Verwechsler (Kalk, Gefaesse, Kavernome) | G |

## 2 Das Original unter der Lupe: MicrobleedNet

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| Sundaresan V et al., "Automated detection of cerebral microbleeds on MR images using knowledge distillation framework", Front. Neuroinform. 17:1204186, 2023 | https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1204186/full | Referenzverfahren; Stichprobe / Verlust (10-fache CMB-Gewichtung), Grenzen laut Autoren (2a, 4.x, `ABWEICHUNGEN-microbleednet.md`) | Q |
| MicrobleedNet-Code, Commit 958f1cb | https://github.com/v-sundaresan/microbleed-detection | Vergleich Paper gegen Code (FRST-Radien, ratio='1:1', Uebermalung); unser Fork `custom_microbleednet` | Q |
| Sundaresan V et al., Eur Radiol Exp 2025 (Groesse und Ort der CMB nach MARS) | https://link.springer.com/article/10.1186/s41747-024-00544-z | MARS-Atlas im MNI-Raum als moegliche Ortsquelle | R |
| Loy G, Zelinsky A, "Fast radial symmetry for detecting points of interest", IEEE TPAMI 25(8):959-973, 2003 | -- | FRST-Kanal (`code/frst.py`; 4.11: ohne FRST -0.053) | G |
| Frangi AF et al., "Multiscale vessel enhancement filtering", MICCAI 1998 | -- | Gefaessfilter der Uebermalung (4.1: gestrichen, +0.136) | G |
| Hinton G, Vinyals O, Dean J, "Distilling the Knowledge in a Neural Network", 2015 | https://arxiv.org/abs/1503.02531 | Destillation im Diskriminator des Originals (nicht nachgebaut) | G |

## 3 Neueste Arbeiten zu CMB (Messlatten und Ideenquellen)

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| He L, Zhang H, Li K, Saccoh AF, Ingala S, Rehwald R, de Bruijne M, Barkhof F, Davies R, Sudre CH, "CenSynCMB: Centre Maps and Physics-Guided Synthesis for Microbleed Detection", arXiv 06.07.2026 | https://arxiv.org/abs/2607.05325 | neue Messlatte auf VALDO Task 2 (74.3); Zentrumskarte (5k), Neugewichtung nach Uebersehenen (5l), Synthese; ihr Auswerteprotokoll in `cmb/analysis/subject_level_f1.py` | Q |
| Tsuchida A et al., SHIVA-CMB, Scientific Reports 2024 | https://www.nature.com/articles/s41598-024-81870-5 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11680838/ ; Code https://github.com/pboutinaud/SHIVA_CMB | fremde Messlatte (5a); trainiert auf 60 der 72 VALDO-Faelle -> fair nur auf catalina | Q |
| "A geometric-to-neural cascade for cerebral microbleed detection in susceptibility-weighted MRI", 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC13483767/ | zwei leichte Klassifikatoren hintereinander, 76.8 % der Fehlalarme entfernt (Motivation zweite Stufe) | Q (Lesemodell) |
| Liu S et al., "Cerebral microbleed detection using Susceptibility Weighted Imaging and deep learning", NeuroImage 198:271-282, 2019 | https://pubmed.ncbi.nlm.nih.gov/31121296/ | 3D-FRST-Kandidaten + ResNet mit Phasenbild (bei uns keine Phase) | R |
| Al-masni MA et al., zweistufig YOLO + 3D-CNN, 2020 | https://pubmed.ncbi.nlm.nih.gov/33018167/ ; https://www.sciencedirect.com/science/article/pii/S2213158220303016 ; Code https://github.com/Yonsei-MILab/Cerebral-Microbleeds-Detection | zweistufige Vergleichsarbeit | R |
| Rashid T et al., "DEEPMIR", Scientific Reports 2021 | https://arxiv.org/abs/2010.00148 | CMB gegen Eisenablagerung (braucht QSM) | R |
| Myung MJ et al., J Stroke Cerebrovasc Dis 2021 | https://www.sciencedirect.com/science/article/abs/pii/S1052305721002895 | Liquor- / Mark-Maske als Ortsmerkmal (vgl. unsere Liquor-Regel 4.3) | R |
| VALDO-Teams: Kuijf H, MixMicrobleedNet (nnU-Net); MixMicrobleed (Mask-RCNN + U-Net); Chen Z ("Zihao", mehrstufig) | https://arxiv.org/abs/2108.01389 ; https://arxiv.org/abs/2108.02482 ; https://github.com/ZihaoChen0319/CMB-Segmentation | was die Challenge-Spitze anders machte (`recherche/valdo-challenge.md`) | R |
| "Toward Automated Detection of Microbleeds with Anatomical Scale Localization" | https://arxiv.org/abs/2306.13020 | Ortswissen gegen Fehlalarme (5i: bei uns ueber die Liquor-Regel hinaus ohne Gewinn) | R |

## 4 Netzarchitekturen des Turniers (4.4)

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| Ronneberger O et al., U-Net, 2015; Cicek O et al., 3D U-Net, 2016 | https://arxiv.org/abs/1505.04597 ; https://arxiv.org/abs/1606.06650 | Grundbauart aller Netze | G |
| Isensee F et al., "nnU-Net", Nature Methods 18:203-211, 2021 | -- | a02-nnunet; anisotrope Kerne / Pooling als Vorbild fuer den Sieger a03-aniso; Deep Supervision | G |
| Oktay O et al., "Attention U-Net", 2018 | https://arxiv.org/abs/1804.03999 | a01-attunet, a11-anisoatt | G |
| Zhou Z et al., "UNet++", 2018 | https://arxiv.org/abs/1807.10165 | a06-unetpp | G |
| Myronenko A, "3D MRI brain tumor segmentation using autoencoder regularization" (SegResNet), 2018 | https://arxiv.org/abs/1810.11654 | a09-segres | G |
| Hatamizadeh A et al., "Swin UNETR", 2022 | https://arxiv.org/abs/2201.01266 | a08-swin (kollabiert, abgebrochen) | G |
| He K et al., ResNet, 2015; Hu J et al., Squeeze-and-Excitation, 2017 | https://arxiv.org/abs/1512.03385 ; https://arxiv.org/abs/1709.01507 | a05-resse | G |
| Ulyanov D et al., Instance Normalization, 2016; Loshchilov I, Hutter F, AdamW 2017 und SGDR (Kosinus) 2016 | https://arxiv.org/abs/1607.08022 ; https://arxiv.org/abs/1711.05101 ; https://arxiv.org/abs/1608.03983 | Trainingsrezept (4.5) | G |

## 5 Vorverarbeitung und Werkzeuge

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| Billot B et al., "SynthSeg", Medical Image Analysis 2023; robuste Variante: Billot B et al., PNAS 2023 | https://arxiv.org/abs/2107.09559 | Gewebekarte: Liquor-Regel (4.3), Ort der Fehler (5i), Gewebemerkmal der zweiten Stufe (5j), `--robust` direkt auf T2* / SWI | G |
| Isensee F et al., "HD-BET", Human Brain Mapping 40:4952-4964, 2019 | -- | Hirnmasken catalina (4.2, 5n: eng gegenueber den lockeren VALDO-Masken) | G |
| Zhang Y, Brady M, Smith S, FSL FAST, IEEE TMI 20:45-57, 2001; Jenkinson M et al., "FSL", NeuroImage 2012 | -- | Biaskorrektur `fast -B` (4.9: ohne -0.022) | G |
| Fischl B, "FreeSurfer", NeuroImage 2012 | -- | Laufzeitumgebung von SynthSeg (FreeSurfer 8.2.0) | G |

## 6 Verluste und Trainingshebel (5g, 5j-5l)

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| Kofler F et al., "blob loss: instance imbalance aware loss functions for semantic segmentation", 2022/2023 (IPMI 2023 (Information Processing in Medical Imaging), Springer LNCS -- Tagung am 23.09.2026 an der Quelle bestaetigt) | https://arxiv.org/abs/2205.08209 | `--verlust blob` (5j): kleine CMB 0.52 -> 0.64, F1 nicht besser | R |
| Milletari F et al., "V-Net" (Dice-Verlust), 2016 | https://arxiv.org/abs/1606.04797 | BCE + Dice im Rezept; Ursache des Kollapses bei 1:2 (5g) | G |
| Abraham N, Khan NM, "A Novel Focal Tversky loss function ...", ISBI 2019 | https://arxiv.org/abs/1810.07842 | `cv5.loss_focal_tversky` (frueher getestet) | R |
| Zhou X et al., "Objects as Points" (CenterNet), 2019; Kendall A et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018 | https://arxiv.org/abs/1904.07850 ; https://arxiv.org/abs/1705.07115 | Hintergrund zur Zentrumskarte und zu ihrer Gewichtung in CenSynCMB (5k) | G |
| Shrivastava A et al., OHEM, 2016; Lin T-Y et al., Focal Loss, 2017 | https://arxiv.org/abs/1604.03540 ; https://arxiv.org/abs/1708.02002 | harte Negative (5l); Focal Loss geprueft und als unpassend verworfen (Kandidaten sind balanciert) | G |
| Weitere Instanz-Verluste aus der Recherche: "Learning to Look Closer" (CC-DiceCE), ICI loss, CATMIL | https://arxiv.org/abs/2511.17146 ; https://arxiv.org/abs/2304.06229 ; https://arxiv.org/abs/2604.08015 | nicht getestet | R |

## 7 Zweite Stufe, Ensemble, Testzeit-Augmentierung (5d, 5j)

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| Setio AAA et al., "Pulmonary Nodule Detection in CT Images: False Positive Reduction Using Multi-View Convolutional Networks", IEEE TMI 35:1160-1169, 2016 | https://pubmed.ncbi.nlm.nih.gov/26955024/ | Vorbild fuer Fehlalarm-Reduktion auf zentrierten Kandidaten | R |
| Ghafoorian M et al., "Deep multi-scale location-aware 3D CNNs for automated detection of lacunes", NeuroImage: Clinical 2017 | https://arxiv.org/abs/1610.07442 | Ort + mehrere Skalen als Eingabe der zweiten Stufe (unser Gewebemerkmal) | R |
| Breiman L, "Random Forests", Machine Learning 2001; Friedman JH, Gradient Boosting, Annals of Statistics 2001 | -- | Klassifikatoren in `cmb/stage2/` | G |
| Zhou Z et al., "Models Genesis", 2019/2021; Chen S et al., "Med3D", 2019 | https://arxiv.org/abs/2004.07882 ; https://arxiv.org/abs/1904.00625 | vortrainierte 3D-Encoder (nicht getestet) | R / G |
| Baumgartner M et al., "nnDetection", MICCAI 2021; Jaeger PF et al., "Retina U-Net", 2018 | https://arxiv.org/abs/2106.00817 ; https://arxiv.org/abs/1811.08661 | Detektionskopf als Alternative (zurueckgestellt) | G / R |
| Lakshminarayanan B et al., "Deep Ensembles", NeurIPS 2017 | https://arxiv.org/abs/1612.01474 | Startwert-Ensemble (`cmb/analysis/ensemble.py`) | G |
| Wang G et al., "Aleatoric uncertainty estimation with test-time augmentation ...", Neurocomputing 2019 | https://www.sciencedirect.com/science/article/pii/S0925231219301961 | Spiegel-TTA (`cmb/analysis/tta_predict.py`) | R |
| Weitere aus der Recherche (nicht gelesen): Zwei-Stufen-System Hirnmetastasen (Eur Radiol 2023), GBT fuer Lungenknoten, Unsicherheit gegen Fehlalarme (Leber), anatomische Nachbearbeitung Aneurysmen | https://pubmed.ncbi.nlm.nih.gov/36695903/ ; https://arxiv.org/abs/1708.05897 ; https://arxiv.org/abs/2206.10911 ; https://arxiv.org/abs/2507.00832 | Ideenquellen, siehe `recherche/zweite-stufe-fp-reduktion.md` | R |

## 8 Uebertragung zwischen Kohorten, Statistik

| Quelle | Link | Wofuer bei uns | Status |
|---|---|---|---|
| "Studying Robustness of Semantic Segmentation under Domain Shift in Cardiac MRI" | https://arxiv.org/abs/2011.07592 | Mischen von Domaenen (5e, 5h, 5n) | R |
| Guan H, Liu M, "Domain Adaptation for Medical Image Analysis: A Survey", IEEE TBME 2022 | -- | Ueberblick (Leseempfehlung, bei uns nicht benutzt) | G |
| Efron B, Tibshirani R, "An Introduction to the Bootstrap", 1993 | -- | gepaarter Bootstrap ueber Faelle (Abschnitt 3) | G |
| Varoquaux G, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage 180:68-77, 2018; Bouthillier X et al., "Accounting for Variance in Machine Learning Benchmarks", 2021 | -- ; https://arxiv.org/abs/2103.03098 | Einordnung unseres Startwert- und Faltenrauschens (Leseempfehlung) | G |

Vollstaendige Recherche-Tabellen mit allen (auch ungeprueften) Treffern: `recherche/valdo-challenge.md`, `recherche/sundaresan-gegen-valdo.md`,
`recherche/datensaetze.md`, `recherche/methoden-segmentierung.md`, `recherche/zweite-stufe-fp-reduktion.md`.
