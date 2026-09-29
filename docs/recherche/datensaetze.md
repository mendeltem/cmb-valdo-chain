# Weitere oeffentliche CMB-Datensaetze, Modelle und Code (Claude, 19.09.2026, sparsame Recherche: 4 Suchen, 3 Abrufe)

Gelesen = Seite selbst abgerufen (von einem kleinen Lesemodell ausgewertet). Treffer = nur in Suchergebnissen gesehen,
NICHT geprueft. Nichts heruntergeladen.

## Datensaetze
| Datensatz | Inhalt | Zugang | Stand |
|---|---|---|---|
| AIBL "Synthetic Cerebral Microbleed on SWI images" (Momeni et al., CSIRO) | 75 SWI mit 175 echten CMB (Orte von Experten), dazu je Scan 10 synthetische CMB (825) | offen, CSIRO Data Licence, DOI 10.25919/AEGY-NY12, https://data.csiro.au/collection/csiro:50304 ; Paper doi 10.1016/J.CMPB.2021.106127 | gelesen; Aufloesung, Dateiformat und ob Masken oder nur Koordinaten: nicht angegeben |
| Dou et al. 2016, "cmb-3dcnn" (CUHK) | 20 SWI (3T Philips) aus einer Studie mit 320; CMB als Koordinaten, etwa 3.7 CMB je Scan | http://www.cse.cuhk.edu.hk/~qdou/cmb-3dcnn/cmb-3dcnn.html | Treffer (in zwei Papern so zitiert); Seite nicht abgerufen. Identisch mit "SHK" bei Sundaresan |
| VALDO Task 2 | 72 T2*-GRE mit Masken | haben wir | -- |
| "Large public dataset of annotated clinical MRIs ... acute stroke" (Sci Data 2023) | 2106 Scans mit SWI, aber Schlaganfall-Annotationen, KEINE CMB-Masken | offen | Treffer; nur als unannotierte SWI (Vortraining, synthetische CMB) brauchbar |
| TICH2, UK Biobank, OXVASC, Rotterdam, MESA ... | gross, mit CMB | nur auf Antrag | nicht frei |

## Vortrainierte Modelle und Code
| Was | Inhalt | Zugang | Stand |
|---|---|---|---|
| SHIVA-CMB (Tsuchida et al., Sci Rep 2024) | 3D-U-Net, trainiert auf 450 Scans aus 7 Akquisitionen / 6 Kohorten, T2*-GRE UND SWI; Test (96 Scans) Sens 0.67, Prec 0.82, F1 0.74, < 1 FP je Bild; Eingabe 160x214x176 | Code https://github.com/pboutinaud/SHIVA_CMB , Gewichte ueber Cloud-Link im Repo; Pipeline https://github.com/pboutinaud/SHiVAi | Treffer (Zahlen aus der Suchzusammenfassung, Lizenz nicht geprueft) |
| Zihao Chen, VALDO-Team "Zihao" | mehrstufig: Kandidaten + Klassifikator (Challenge-Platz 2-3, Median-F1 0.667) | https://github.com/ZihaoChen0319/CMB-Segmentation | Treffer |
| MixMicrobleed (VALDO-Team) | Mask-RCNN + U-Net | arXiv 2108.02482 | Treffer |
| Yonsei-MILab Cerebral-Microbleeds-Detection (Al-masni) | YOLO + 3D-CNN | https://github.com/Yonsei-MILab/Cerebral-Microbleeds-Detection | Treffer |

## Neue Methodenarbeiten, die zu unseren offenen Punkten passen
- "A geometric-to-neural cascade ..." (2026, PMC13483767; gelesen): unueberwachte Kandidaten (GMM-Schwelle + anatomische
  Maske), dann zwei leichte 3D-ResNets; 10 eigene + 20 Dou-Faelle; Sens 0.71, Prec 0.68, F1 0.69; die Klassifikatoren
  entfernen 76.8 % der Fehlalarme. Stuetzt unser geplantes Netz 10 (zweite Stufe).
- "CenSynCMB: Centre Maps and Physics-Guided Synthesis ..." (arXiv 2607.05325; Treffer): Zentrumskarten statt Masken UND
  physikalisch begruendete synthetische CMB -- genau unsere Ideen 4 und 5; vor dem Bau lesen.
- "Learning to Look Closer: A New Instance-Wise Loss for Small Cerebral Lesion Segmentation" (arXiv 2511.17146; Treffer).
- ESOC 2026 Abstract A809: 3D-Attention-U-Net auf SWI mit MARS-Regionen (Treffer).

## Einschaetzung
Frei und sofort nutzbar sind nur AIBL (75 SWI / 175 CMB) und Dou (20 SWI) -- beides SWI, beides vermutlich Punkt- statt
Maskenannotation. Fuer ein T2*-Modell taugen sie als Zusatz-Vortraining und als fremde Testkohorte, nicht als Ersatz.
Der groessere Hebel ist SHIVA-CMB: ein frei verfuegbares Modell, das auf 450 Scans beider Sequenzen trainiert wurde --
als Messlatte auf VALDO und catalina (wie gut ist "von der Stange"?) und als moeglicher Startpunkt fuers Feintuning.

## Nachtrag 19.09. abends: SHIVA-CMB gemessen
Lizenz gelesen (CC BY-NC-SA 4.0), Gewichte v2 geladen (SHA-256 stimmt). Auf unseren 57 VALDO-CV-Faellen F1 0.575 /
Median 0.667 -- ABER laut Paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC11680838/) wurde SHIVA auf 60 der 72
oeffentlichen VALDO-Faelle trainiert (dazu DOU 17, AIBL 44 echt + 60 synthetisch, BBS 269). Fair ist nur catalina.
Dasselbe Paper zeigt auch: die freien Datensaetze DOU und AIBL sind als Trainingsdaten brauchbar (dort so verwendet).
