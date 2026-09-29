#!/usr/bin/env python
"""06 ensemble + score: two-seed ensemble, raw lesion F1 at the fixed cell (0.3 / 2 mm3), then the SynthSeg CSF rule -> 0.647 (deliverable)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["ensemble", "evaluate-basis", "csf"], __doc__)
