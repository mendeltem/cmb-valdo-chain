#!/usr/bin/env python
"""10 baseline: the original MicrobleedNet fine-tuned on the same folds with its own preprocessing, scored with the same metric."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmb.experiments.pipeline import run
run(["mbnet-prepare", "mbnet-finetune", "mbnet-score"], __doc__)
