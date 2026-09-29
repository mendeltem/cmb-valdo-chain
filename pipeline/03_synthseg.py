#!/usr/bin/env python
"""03 SynthSeg: FreeSurfer mri_synthseg tissue maps per case, resampled to the grid (dev/synthseg). Used by the CSF rule and the tissue feature of the classifier."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["synthseg"], __doc__)
