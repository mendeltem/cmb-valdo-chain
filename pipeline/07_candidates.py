#!/usr/bin/env python
"""07 candidates: 16-mm cubes (image, FRST, probability) around every component at 0.15 / 1 mm3 with --keep-csf; input of the second stage."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["candidates-valdo"], __doc__)
