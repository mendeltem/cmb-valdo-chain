#!/usr/bin/env python
"""01 preprocess: brain mask + FSL FAST -B once per VALDO case (dev/preproc), then the inverted image WITHOUT vessel inpainting (dev/preproc_roh). Finished cases are skipped, so HD-BET/bias correction never run twice."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["preprocess", "preproc-roh"], __doc__)
