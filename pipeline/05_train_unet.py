#!/usr/bin/env python
"""05 train: anisotropic 3D U-Net a03-aniso, 60 epochs, 5-fold CV, seeds 42 and 1 (GPU, ~1.6 h each). Deterministic per seed."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["train-basis", "train-s1"], __doc__)
