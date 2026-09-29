# Leichte Nachbearbeitung gegen Gefäß-Fehlalarme: was die Literatur hergibt (29.09.2026; Unteragent sonnet, 9 Suchen; Zitate prüfen)

| # | Verfahren | Quelle | Daten | Zahl | übertragbar auf 2-5 mm? |
|---|---|---|---|---|---|
| 1 | 3D-Radialsymmetrie, halbautomatisch | Kuijf 2013, PLoS ONE | 3 T T2*, 72 Pat. | Sens 65-84 %, 20-96 FP je Fall (dann manuell) | unklar |
| 2 | 2D-FRST auf Minimum-Intensity-Projektion + Region-Growing + Formmerkmale | Bian 2013, NeuroImage Clin 2:282 | SWI (Bestrahlungs-CMB) | Sens 86,5 % bei 44,9 FP je Fall | plausibel (mIP quer zu z) |
| 3 | Schwelle + SVM auf Form/Intensität | Barnes 2011, NeuroImage | 7 T | Sens 81,7 %, ~107 FP je Fall | nein |
| 4 | msLoG-Kandidaten + Radon/Hesse-Deskriptoren + Kaskaden-RF | Fazlollahi 2015, CMIG 46:269 | SWI | Sens 87 %, FP-Zahlen nicht gefunden | fraglich (Hesse bei uns verworfen) |
| 5 | **Voxel-RF (12 Merkmale) -> Objekt-RF mit 7 Formmerkmalen (Elongation gegen Kugel)** | van den Heuvel 2016, NeuroImage Clin, PMID 27489772 | SWI + T1, SHT | **Sens 89,1 % bei 25,9 FP je Fall** | mittel: Objektgeometrie, nicht Krümmung |
| 6 | **Bounding-Box-Achsverhältnis > 3 = Gefäß, Volumen, Diagonale <= 10 mm, dann kleines Netz** | "geometric-to-neural cascade" 2026, PMC13483767 | 1 mm isotrop | Kaskade gesamt: FP 1157 -> 269 (-77 %), Sens 0,93 -> 0,71 | nicht belegt für dicke Schichten |
| 7 | Venenatlas / Frangi-Maske als Ausschluss | -- | -- | **kein Beleg gefunden** | -- |
| 8 | Kalk gegen Blut per QSM-Vorzeichen | Eur Radiol 2024, 10.1007/s00330-024-10889-z | Phase nötig | keine Zahl im Auszug | nur mit Phase |
| 9 | Mimic-Orte: Basalganglien-/Plexus-/Pineal-/Falx-Kalk | mehrfach genannt | -- | keine FP-Zahl ohne QSM | -- |
| 10 | 2D-Merkmale je Schicht + Zahl belegter Schichten + Zentroid-Verschiebung | Prinzip in mehreren Pipelines | -- | keine Einzelzahl | ja, in Minuten prüfbar |

## Die zwei billigsten, noch nicht gemessenen Kandidaten (vom Agenten; heute nachgemessen, `geom_*.json`)
(a) **Achsverhältnis der Kandidatenkomponente** (PCA längste/kürzeste Hauptachse in mm, Schwelle um 3; globale Objektgeometrie, nicht
    lokale Krümmung wie Frangi/Hesse), dazu das In-plane-Verhältnis je Schicht;
(b) **Formstabilität über Schichten**: Zahl der belegten nativen Schichtblöcke und die Schwerpunkt-Verschiebung zwischen Nachbarschichten
    (zielt auf Geometrie, nicht auf dunkle Fortsetzung wie die verworfene Laufstrecke).
Vorbehalt vorab: die Komponenten sind die des U-Nets (Schwelle 0,15), nicht Segmentierungen des Bildes -- ein Gefäß, das das U-Net nur
punktweise markiert, hat eine kompakte Komponente; und die z-Ausdehnung der Komponente war bei uns für ECHTE Blutungen größer (Blockfüllung).
