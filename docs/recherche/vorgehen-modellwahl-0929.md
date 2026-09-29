# Wie die Literatur Modelle wählt und validiert -- und was wir davon schon tun (29.09.2026; Unteragent sonnet, 10 Suchen; Zitate vor Gebrauch prüfen)

| # | Aussage | Quelle | Bei uns |
|---|---|---|---|
| 1 | nnU-Net: feste Heuristik statt Hyperparametersuche, 5-fach-CV | Isensee 2021, Nat. Methods 18 | Turnier 2 ist genau diese Art: je eine gezielte Änderung, kein Grid/Bayes |
| 2 | "nnU-Net Revisited": vier Fallen -- Neuerung nicht vom Confounder getrennt, keine Standard-Baseline, zu wenige/ähnliche Datensätze, uneinheitliches Berichten | Isensee 2024, arXiv 2404.09556 | Baseline vorhanden (a02 DynUNet 0.458 sauber; ResEnc eingereiht), aber nicht das nnU-Net-Framework mit eigener Vorverarbeitung -- als Grenze nennen; drei Kohorten + Momeni = vier Datensätze |
| 3 | Metrics Reloaded: naives Pooling über hierarchische Daten bevorzugt läsionsreiche Fälle; je Fall mitteln | Maier-Hein 2024, Nat. Methods 21 | gepoolt ist unser Hauptmaß (sub-110 trägt 75 % der Prüfmenge!). Bootstrap läuft schon je Fall; zusätzlich Mittel/Median je Fall als zweites Hauptmaß berichten (subject_level_f1 liefert beides) |
| 4 | Instanzmetrik: Lokalisationskriterium und Zuordnung VORAB festlegen | dieselbe, Fig. 2 | steht: 26er-Nachbarschaft, Berührungsregel, Riesenkomponenten weg (ARBEIT.md 4.3, REPLIKATION 4); Challenge-Regel (Zentrum 5 mm) daneben |
| 5 | Datensampling ist die größte Varianzquelle, Init < 50 %; viele Wiederholungen nötig | Bouthillier 2021, MLSys | unser Fall-Bootstrap deckt das Sampling; Startwert-Streuung ist gemessen (VALDO 7 Startwerte 0.569-0.646 mit Regel; zweite Stufe 10 Startwerte sd 0.003-0.009) |
| 6 | Seeds erzeugen Ausreißer, Varianz sinkt nicht mit längerem Training | Picard 2021 | deshalb Regel: ein Startwert entscheidet nichts unter dem Startwert-Maximum der Basis (0.646); Turnier-2-Schwelle +0.04 liegt darüber |
| 7 | HPO bei < 100 Fällen: kein Beleg für Nutzen; nnU-Net meidet es bewusst | Isensee 2021; Auto-nnU-Net 2025 | Turnier 2 dient dem Beleg der Suche, nicht der Hoffnung |
| 8 | VALDO-Übersicht: Dice + Volumetrie als Challenge-Maß, große Streuung, "Lücke auf Fallebene" | arXiv 2208.07167 | unser Läsions-F1 + gepaarter Bootstrap geht darüber hinaus |
| 9 | MicrobleedNet: intern Dice 0.73, extern 0.46 -- der Domänensprung ist größer als jedes Seed-Rauschen | PMC12091992 (2024/25) | gemessen: VALDO-Modell auf catalina-SWI 0.305, auf Momeni 0.613; eigenes Training je Kohorte ist die Antwort |
| 10 | MixMicrobleedNet: fold=all, Selbsttest auf Trainingsdaten | arXiv 2108.01389 | wir: 5-fach-CV, Prüfmenge einmal |
| 11 | SHIVA-CMB: 450 Scans, 7 Akquisitionen, gehaltener Testsatz + externe Kohorte (F1 0.74, S 0.67, P 0.82) | Boutinaud 2024, Sci Rep | wir haben den externen Test (Momeni) -- aber nur 57 Fälle einer Akquisition; Trainingsbreite (7 Akquisitionen) fehlt uns |
| 12 | CenSynCMB: fold-weise leckagefrei, Streuung über Folds + p-Werte, externe Kohorte als zusätzliche Kontrolle | He 2026, arXiv 2607.05325 | deckungsgleich mit unserem Design; Folds-Streuung berichten wir (F1 je Falte in Rangliste) |

## Was wir daraus ändern (Vorschlag)
1. **Zweites Hauptmaß je Fall** (Mittel und Median des F1 je Fall, wie Metrics Reloaded) neben dem gepoolten F1 in JEDER Tabelle -- billig, Werkzeug vorhanden. Damit ist die sub-110-Dominanz methodisch adressiert.
2. **Startwert-Nullverteilung der Basis explizit als Tabelle** (VALDO: s42 0.646, s1 0.590, s2 0.578, s3 0.603, s4 0.608, s5 0.569, s6 0.618; s7-s9 noch auswerten): jeder Einzelstartwert-Lauf des Turniers wird gegen diese Verteilung gelesen, nicht nur gegen s42.
3. **Stichprobenartig zweite Startwerte auch für Verlierer** (Picard): zwei zufällig gewählte Turnier-2-Läufe bekommen einen zweiten Startwert, damit die Schwelle +0.04 nicht nur Gewinner absichert.
4. **nnU-Net-Framework als Baseline** (Revisited): das echte nnU-Net v2 mit eigener Vorverarbeitung auf VALDO, ein Lauf -- die einzige Lücke, die ein Gutachter sicher nennt. Aufwand: Installation + ~6 h GPU (1000 Epochen sind nnU-Net-Standard).
5. Grenzen im Paper benennen: eine Akquisition je Kohorte, Momeni als einzige externe Kohorte.
