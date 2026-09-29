#!/usr/bin/env python
"""08 second stage: pooled 3D-CNN candidate classifier over the cohorts (needs the private candidate sets) and the VALDO-only transfer variant. Without private data use: python -m cmb.stage2.apply --source ergebnisse/stage2/a03-aniso-e60-gd-weich --target NAME=<candidates>."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["stage2-pooled", "stage2-transfer"], __doc__)
