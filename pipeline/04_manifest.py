#!/usr/bin/env python
"""04 manifest: case list of the 57 CV cases with fold, grid paths and lesion counts (dev/manifest_valdo_gitter_d.json). The 15 held-out cases are excluded here and were measured once (see docs/WORKLOG.md 5ad)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["manifest"], __doc__)
