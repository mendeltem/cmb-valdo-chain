# cmb -- clean English package for the microbleed experiments

The scripts in `../code` grew during exploration: German names, dense one-liners, behaviour switched by
monkeypatching. This package replaces them module by module. **Rule: a module is accepted only if it reproduces
its predecessor exactly** (the pipeline is deterministic: same seed -> same F1 to four decimals; networks must
load the saved tournament checkpoints and give bit-identical outputs).

The old scripts stay untouched while the GPU queue is calling them.

| Module | Replaces | Status |
|---|---|---|
| `models/blocks.py` | helpers in `netze.py`, `attention_unet.py` | done |
| `models/anisotropic_unet.py` | `AnisoUNet3D` (`a03-aniso`, `a11-anisoatt`) | done, bit-identical (test below) |
| `models/legacy.py` | -- | done: loads old `modell.pt` files |
| `models/*` remaining 7 tournament nets | `netze.py` | next |
| `data.py` (loading, SynthSeg mask, z-normalisation) | `cv5.lade_daten`, wrappers in `turnier.py` | planned |
| `sampling.py` (patch sampling, augmentation) | `cv5.stichprobe`, `mit_augmentierung` | planned |
| `losses.py`, `inference.py` (sliding window), `postprocess.py` (size filter, CSF rule) | `cv5.py`, `gewebekarte.py` | planned |
| `metrics.py` | `metric.py` (touch rule), matching half of `metric_valdo.py` | **done, verified identical** (`cmb.tests.test_metrics_match_legacy`: 40 random masks, 25-case summary, 30 VALDO matches, 8 real cases). Two stated differences: f1 unrounded, and the diagnostic extras of the old file are not carried over -- `cv5_agg.py` and `fix_block_we4.py` still use them, so the old file stays |
| `train.py` (one fold), `cli/` (tournament, ranking, tissue map, vessel inpainting) | `cv5.train_falte`, `turnier*.py`, `uebermalung2.py` | planned |
| `preprocessing/vessel_inpainting.py` | `uebermalung2.py` | planned |

Tests (run from `valdo-t2s/`): `python -m cmb.tests.test_models_match_legacy`, `python -m cmb.tests.test_metrics_match_legacy`.

**Why the rest is not ported yet (2026-09-23):** `turnier.py` and `cv5.py` are the two largest modules (843 and 478
lines) and are being called by the GPU queue right now; porting them means touching running scripts. They come once
the queue is empty. The leaf modules (`metrics`, then `frst`, `gewebekarte`) can be done at any time because nothing
running depends on them changing.

Style: English everywhere; one job per module; type hints; docstrings state tensor shapes; comments explain WHY
(with the measurement that motivated the choice); explicit configuration objects instead of patched globals;
clear error messages.
