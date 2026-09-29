"""The ported networks must be the SAME function as the old ones.

For each ported architecture: load the weights of a finished tournament fold into both
implementations and compare the outputs on a random patch of the real (odd) patch size.

    python -m cmb.tests.test_models_match_legacy          (run from valdo-t2s/)
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))          # old scripts
sys.path.insert(0, ROOT)                                 # this package

from cmb.models import build_model, count_parameters    # noqa: E402
from cmb.models.legacy import load_legacy_checkpoint     # noqa: E402
import netze as legacy_models                            # noqa: E402

PATCH = (38, 62, 94)      # (z, y, x) voxels, the training patch size


def check(legacy_name: str) -> None:
    checkpoint = os.path.join(ROOT, "ergebnisse", "turnier", legacy_name, "fold0", "modell.pt")
    old = legacy_models.NETZE[legacy_name](2).eval()
    old.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    new = load_legacy_checkpoint(build_model(legacy_name, in_channels=2), checkpoint).eval()

    torch.manual_seed(0)
    x = torch.randn(1, 2, *PATCH)
    with torch.no_grad():
        difference = (old(x) - new(x)).abs().max().item()
    assert count_parameters(old) == count_parameters(new)
    assert difference == 0.0, f"{legacy_name}: outputs differ by {difference}"
    print(f"{legacy_name:14s} ok  ({count_parameters(new) / 1e6:.2f} M parameters, max |difference| = {difference})")


if __name__ == "__main__":
    for name in ("a03-aniso", "a11-anisoatt"):
        check(name)
