"""``cmb selftest`` (no data, CPU, about 1 min): imports, FRST and lesion metric self-checks of the paper code, one
forward pass of the a03-aniso U-Net, and the paper's grid builder on a synthetic phantom. Ends with SELFTEST OK."""
from __future__ import annotations

import os
import sys
import tempfile
import time

from cmb.paths import ROOT

ok = True


def check(name, fn):
    global ok
    t = time.time()
    try:
        fn()
        print(f"ok    {name} ({time.time() - t:.1f} s)", flush=True)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)


def imports():
    import numpy, scipy, nibabel, skimage, sklearn, SimpleITK, torch  # noqa: F401
    import cmb.transfer.evaluate, cmb.analysis.operating_point, cmb.stage2.candidates, cmb.stage2.pooled  # noqa: F401
    import cmb.stage2.trackset, cmb.training.full_data, cmb.predict_cli, cmb.commands, cmb.api  # noqa: F401
    print(f"      torch {torch.__version__}, cuda {'yes' if torch.cuda.is_available() else 'no'}")


def frst_test():
    import frst
    frst.selftest()


def metric_test():
    import metric
    metric.selftest()


def net_forward():
    import torch
    import netze
    net = netze.NETZE["a03-aniso"](2).eval()
    with torch.no_grad():
        y = net(torch.zeros(1, 2, 16, 64, 64))
    assert tuple(y.shape[-3:]) == (16, 64, 64), y.shape
    print(f"      a03-aniso: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters")


def grid_test():
    import nibabel as nib
    import numpy as np
    from cmb import predict_cli as predict, synthetic
    img, lab, brain, aff = synthetic.phantom(0)
    with tempfile.TemporaryDirectory() as d:
        ni = nib.Nifti1Image(img, aff)
        info = predict.build_grid("case", predict.invert(img), ni, d)
        g = nib.load(f"{info['dir']}/case_frst.nii.gz")
        assert np.isfinite(np.asarray(g.dataobj)).all() and info["affinen_konsistent"], info
        print(f"      grid {info['quelle_shape']} -> {info['shape']} (0.5 x 0.5 x 1 mm)")


def main():
    sys.path[:0] = [os.path.join(ROOT, "code")]
    for n, f in [("FRST", frst_test), ("lesion metric", metric_test), ("U-Net forward", net_forward), ("imports", imports),
                 ("grid builder on a phantom", grid_test)]:
        check(n, f)
    print("SELFTEST OK" if ok else "SELFTEST FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
