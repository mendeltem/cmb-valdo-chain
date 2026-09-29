# Hat MicrobleedNet von VALDO gelernt? Und was lernen wir von beiden? (Claude, 19.09.2026, sparsame Recherche)

Quellen: Sundaresan et al., Front. Neuroinform. 2023 (https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1204186/full),
von einem kleinen Lesemodell ausgewertet, Zitate woertlich uebernommen, von mir nicht zeilenweise gegengelesen;
VALDO: recherche/valdo-challenge.md; Code: Commit 958f1cb, selbst gelesen. Sundaresan et al. 2025 (Eur. Radiol. Exp.,
https://link.springer.com/article/10.1186/s41747-024-00544-z) nur aus der Kurzfassung.

## Gelesen
- Zeitlich: MicrobleedNet-Preprint medRxiv November 2021, VALDO-Challenge MICCAI 2021 (Paper 2022/2024) -- parallel entstanden.
- Im Paper 2023 wird VALDO NUR benutzt: "20 subjects for hyper-parameter tuning". Kein Training, kein Test auf VALDO,
  kein Vergleich mit den VALDO-Teams, keine ihrer Methoden zitiert. Daten: UK Biobank (78, SWI/QSM, 0.8x0.8x3 mm),
  OXVASC (74, T2*-GRE, 0.9x0.8x5 mm), TICH2 (115, SWI), SHK (20, SWI).
- Ergebnisse dort (andere Daten, andere Metrik als unsere): Cluster-TPR 0.90-0.93 bei 0.9-1.5 FP je Fall in der CV;
  ueber Datensaetze hinweg TPR 0.81-0.87, Precision 0.44-0.89.
- Eingestandene Grenzen (woertlich): Drehung und Verkleinerung koennen CMB verlieren; die Gefaessentfernung kann echte
  CMB nahe an Gefaessen entfernen; Schwellen wurden je Datensatz empirisch gesetzt; 5-mm-Schichten machen
  Partialvolumen-Probleme; auf T2*-GRE ist der Kontrast schwaecher als auf SWI.
- Paper gegen Code (selbst gelesen): FRST laut Paper Mittel ueber vier Radien 2/3/4/6 Voxel -- im Code `radii = [2, 3]`.
  Nachbearbeitung laut Paper: Volumen < 2.5 mm3 weg, Elliptizitaet > 0.2 weg, < 5 mm vom Maskenrand weg; im Code noch
  nicht nachgelesen.
- 2025 (nur Kurzfassung): keine neue Detektion, sondern Groesse und Ort der CMB nach MARS (infratentoriell / tief /
  lobaer) ueber Atlanten; Atlas der MARS-Regionen im MNI-Raum ist oeffentlich. Ob VALDO vorkommt: nicht gefunden.

## Antwort
Nach dem, was im Paper steht: MicrobleedNet hat VALDO als Daten zum Einstellen genutzt, aber aus den ERGEBNISSEN der
Challenge nichts uebernommen -- konnte es zeitlich auch kaum. Umgekehrt nutzte keines der VALDO-Teams FRST,
Gefaessentfernung oder Destillation.

## Was wir von beiden uebernehmen (interpretiert)
| Lehre | Von wem | Bei uns |
|---|---|---|
| Langes Training, grosser Kontext, T1+T2+T2* | VALDO-Spitze | 60 -> 300 Epochen laeuft; Patchgroesse und T1/T2-Kanaele stehen an |
| 3D gegen Mimics; 2D faellt ab | VALDO | bestaetigt: unser 2.5D-Netz nur Platz 2, nicht gesichert |
| Zweistufig (Kandidaten + Klassifikator) ist gleichauf mit der Spitze | beide | Netz 10 noch zu bauen |
| Anisotropie ernst nehmen | nnU-Net (VALDO-Sieger) waehlt anisotrope Kerne automatisch | unser gesicherter Gewinn +0.16 |
| FRST als Wegweiser-Kanal | Sundaresan | behalten; ohne Uebermalung 96 % der CMB im obersten 1 % |
| Randabstand und Formfilter als Nachbearbeitung | Sundaresan | bei uns nie gemessen -> billige Gittersuche auf vorhandenen Karten |
| Rotation/Skalierung kann CMB zerstoeren | beide warnen | passt zu unserem negativen Augmentierungs-Ergebnis (-0.06) |
| Gefaessentfernung kann CMB entfernen | Sundaresan nennt es selbst als Grenze | von uns gemessen: 109 von 236 getroffen |
| Ort der CMB nach MARS mit oeffentlichem Atlas | Sundaresan 2025 | besser als meine SynthSeg-Naeherung, die tiefes Marklager nicht trennen kann |
| Schwellen je Datensatz empirisch | Sundaresan | bei uns ueber die innere Falte -- fuer catalina neu bestimmen, nicht uebernehmen |
