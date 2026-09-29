#!/usr/bin/env python
"""02 grid + FRST: resample to 0.5 x 0.5 x 1 mm (dev/gitter_d) and compute the fast radial symmetry transform channel. Run pipeline/00_selftest.py first: it checks the FRST on synthetic discs and lines."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["grid"], __doc__)
