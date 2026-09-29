# VALDO-Challenge, Task 2 (CMB) -- was das Paper sagt (Claude, 18.09.2026, sparsame Recherche)

Quelle: Sudre et al., "Where is VALDO?", arXiv 2208.07167 (HTML-Fassung https://arxiv.org/html/2208.07167),
zwei gezielte Abrufe; ausgelesen von einem kleinen Lesemodell, Tabellenwerte woertlich uebernommen,
von mir NICHT Zeile fuer Zeile gegengelesen. Auswertungscode: https://github.com/WhereIsValdo/valdo-eval-2021
(README ohne Metrikdetails; Quelltext nicht gelesen).

## Gelesen
- Metrik Task 2: F1 und absolute Differenz der Anzahl (Detektion), mittlerer Dice und Volumendifferenz.
  Zuordnung: Schwerpunkt der Vorhersage < 5 mm vom Schwerpunkt der Referenz; Komponenten mit
  6er-Nachbarschaft, Schwelle 0.5. Berichtet wird der MEDIAN JE PROBAND [Quartile], kein gepooltes F1,
  keine Sensitivitaet/Precision/FP je Scan. Testsatz: 147 (SABRE 20, RSS 68, ALFA 59), verdeckt.
- Inter-Rater-Uebereinstimmung fuer CMB: im Paper NICHT berichtet. Referenz: BOMBS (SABRE, ALFA) bzw.
  Vernooij 2008 (RSS); bei zwei Annotationen deren Mittel.
- Ergebnisse (Median-F1 je Proband in %, Median-Dice): MixMicrobleedNet 68.4 / 84.0 (nnU-Net 3D,
  ganzes Bild, alle Modalitaeten, 1000 Epochen); TeamTea 66.7 / 82.6 (3D-U-Net, Patch 96x192x128,
  Biasfeldkorrektur, alle Modalitaeten); Zihao 66.7 / 80.0 (ZWEISTUFIG: 3D-U-Net-Kandidaten +
  FCN/AlexNet-Klassifikator auf 20x20x16 / 24x24x20, wBCE+Dice); ValdoNN 50.0 (nnU-Net 2D);
  Tfff 40.0 (2D); BigrBrain 16.7 (nnU-Net 2D); Dawai 0.0 (3D-U-Net mit blob loss); TheGPU 0.0
  (Random Forest, nur T2*). Je Kohorte nur fuer zwei Teams: Zihao ALFA 50 / RSS 74.8 / SABRE 45.
- ALLE Deep-Learning-Teams nutzten T1 + T2 + T2* (im T2*-Raum). Nur das Random-Forest-Team nahm T2* allein.
- Autoren: 3D-Information "particularly relevant to avoid mimics"; selbst die Besten uebersehen Faelle
  und melden CMB in Faellen ohne CMB; Augmentierung mit Interpolation/Deformation kann schaden.

## Nicht gefunden
Wie F1 bei Probanden OHNE Referenz-CMB gerechnet wird; wie viele Testprobanden ohne CMB; Details der
Siegermethode (Nachbearbeitung, Umgang mit Negativfaellen); worauf sich "use of the whole extent of
the training data" genau bezieht.

## Folgerungen fuer uns (interpretiert)
1. Unsere 0.25-0.33 (GEPOOLT, 26er-Beruehrung, 57 CV-Faelle) sind mit den 0.67 (Median je Proband,
   5-mm-Schwerpunkt, verdeckter Test) NICHT direkt vergleichbar. Erst mit code/metric_valdo.py den
   Median je Fall rechnen, dann vergleichen.
2. Die drei Besten unterscheiden sich von uns weniger in der Architektur als in: (a) Kontext -- ganzes
   Bild bzw. 96x192x128 gegen unsere 38x62x94 Voxel (47x31x38 mm); (b) Trainingsdauer -- nnU-Net
   1000 Epochen gegen hoechstens 40 x 800 Patches; (c) Eingaenge -- T1+T2+T2* gegen T2*+FRST.
3. Zweistufig (Zihao) liegt gleichauf mit dem Sieger -> Netz 10 ist gerechtfertigt.
4. 2D schneidet durchweg schlechter ab (16.7 / 40 / 50) -> a04-2p5d wird vermutlich nicht gewinnen.
5. blob loss brachte dort nichts (Dawai 0.0) -- kein Grund, ihn vorzuziehen.
