#!/usr/bin/env python
"""11 quality control: slice viewer (reference left, model outlines right, scroll = z) for the VALDO results -> ergebnisse/qc/valdo_slices; the public copy is https://mendeltem.github.io/valdo-cmb-qc/slices/"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["qc-viewer"], __doc__)
