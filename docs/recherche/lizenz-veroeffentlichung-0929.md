# Lizenz für Code, Gewichte und QC-Seiten (29.09.2026; Unteragent sonnet, 6 Suchen + 2 Seiten; KEINE Rechtsberatung, Unsicheres markiert)

| Frage | Befund | Quelle |
|---|---|---|
| CC BY-NC-SA 4.0 und trainierte Gewichte | "Adapted Material" = Bearbeitung, die urheberrechtlich Erlaubnis braucht; ob Gewichte darunter fallen, ist **nicht entschieden** (hängt von der Rechtsordnung ab). Falls ja: ShareAlike, also Gewichte ebenfalls BY-NC-SA. Code ist davon unabhängig und darf MIT/Apache sein. | creativecommons.org FAQ; CC-Papier "Using CC-licensed Works for AI Training" (2025); wiki.creativecommons.org/wiki/4.0/Treatment_of_adaptations |
| MicrobleedNet (v-sundaresan/microbleed-detection) | keine LICENSE-Datei erkennbar; Gewichte per Google-Drive ohne Bedingungen | github.com/v-sundaresan/microbleed-detection |
| SHIVA-CMB | **Code AGPLv3, Gewichte CC BY-NC-SA** -- das direkte Vorbild im Fach | github.com/pboutinaud/SHIVA_CMB, /SHiVAi; Sci Rep 2024 |
| TotalSegmentator | Code Apache-2.0; einzelne Modelle auf restriktiven Daten CC BY-NC(-SA) | github.com/wasserth/TotalSegmentator |
| VALDO Task 2 | Zenodo 4520773, CC BY-NC-SA 4.0, Zitier- und Danksagungspflicht (Förderer, "Where is VALDO", Med Image Anal 2024); Kohorten SABRE, Rotterdam, ALFA | zenodo.org/records/4520773; valdo.grand-challenge.org/Task2 |
| Momeni/CSIRO | CSIRO Data Licence: nicht-kommerziell, KEINE Weitergabe der Rohdaten; erlaubt "hardcopy products and non-editable digital images (z. B. PDF)" kostenlos -- statische Ausschnitte tendenziell ja, interaktiver Voxel-Betrachter **unklar** | research.csiro.au/dap/licences/csiro-data-licence |
| CenSynCMB | keine belastbare Lizenzangabe gefunden | -- |

## Empfehlung fürs Repo (vorsichtige Lesart, keine Rechtsaussage)
1. Zwei Lizenzdateien: `LICENSE` (Code) = MIT oder Apache-2.0; `LICENSE-WEIGHTS` = CC BY-NC-SA 4.0 mit dem Satz "Trained on VALDO Task 2
   (CC BY-NC-SA 4.0, non-commercial), zenodo.org/records/4520773; cite Sudre et al. 2024" -- wie SHIVA-CMB. Formulierung: "aus Vorsicht
   BY-NC-SA gewählt", nicht als sichere Rechtsfolge.
2. QC-Seite mit VALDO-Ausschnitten: nicht-kommerziell + Quellenangabe sichtbar -> gleiche Logik wie die Daten selbst (Repo valdo-cmb-qc gibt es schon).
3. Momeni-Ausschnitte: nur statisch (PNG/PDF), mit Quelle, ohne Umsatz; den Schichtbetrachter mit Momeni-Bildern NICHT veröffentlichen
   (Voxeldaten liegen als JPEG-Stapel darin -- das ist "editable"-nah). Die Zahlen (F1 etc.) sind unproblematisch.
4. catalina: nie, auch nicht als Ausschnitt (Patientendaten, eigene Regel).
5. Vor der Veröffentlichung: Zitierpflichten von VALDO (Förderer-Danksagung) in README und Paper übernehmen.
