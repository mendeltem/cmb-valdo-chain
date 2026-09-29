#!/usr/bin/env python
"""00 self-test (no data, CPU, < 1 min): imports, FRST on synthetic discs and lines, lesion metric on synthetic balls,
one forward pass of the a03-aniso U-Net, and a dry run of every pipeline step. Run this first after installing."""
import os, sys, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [R, os.path.join(R, "code")]
ok = True
def check(name, fn):
    global ok
    t = time.time()
    try:
        fn(); print(f"ok    {name} ({time.time()-t:.1f} s)")
    except Exception as e:                       # noqa: BLE001
        ok = False; print(f"FAIL  {name}: {type(e).__name__}: {e}")

def imports():
    import numpy, scipy, nibabel, skimage, torch, monai   # noqa: F401
    import cmb.transfer.evaluate, cmb.analysis.operating_point, cmb.stage2.candidates, cmb.stage2.pooled, cmb.review.build_slice_viewer  # noqa: F401
    print(f"      torch {torch.__version__}, monai {monai.__version__}, cuda {'yes' if torch.cuda.is_available() else 'no (training needs a GPU)'}")
def frst_test():
    import frst; frst.selftest()
def metric_test():
    import metric; metric.selftest()
def net_forward():
    import torch, netze
    net = netze.NETZE["a03-aniso"](2).eval()
    with torch.no_grad():
        y = net(torch.zeros(1, 2, 16, 64, 64))
    assert tuple(y.shape[-3:]) == (16, 64, 64), y.shape
    print(f"      a03-aniso: {sum(p.numel() for p in net.parameters())/1e6:.1f} M parameters, output {tuple(y.shape)}")
def dry_run():
    from cmb.experiments import kern
    sys.argv = ["kern", "--all", "--dry-run", "--force"]; kern.main()

for n, f in [("imports", imports), ("FRST", frst_test), ("lesion metric", metric_test), ("U-Net forward", net_forward), ("dry run of all steps", dry_run)]:
    check(n, f)
print("SELFTEST", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
