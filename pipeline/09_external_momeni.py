#!/usr/bin/env python
"""09 external validation: public Momeni/CSIRO SWI cohort (grid, VALDO U-Net zero-shot, SynthSeg, CSF rule) -> 0.613."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["momeni-grid", "momeni-zeroshot", "momeni-synthseg", "momeni-score"], __doc__)
