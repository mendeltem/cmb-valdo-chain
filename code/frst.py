#!/usr/bin/env python3
"""FRST channel, new version (Claude 11 Sep 2026): Fast Radial Symmetry Transform
after Loy & Zelinsky (2003, IEEE PAMI 25(8)), per axial slice, vectorized.

Why new: the old implementation (microbleednet, scripts/data_preparation.py
fast_radial_symmetry_xfm, ~line 530-590) has three bugs that distort the
signal -- measured in the --selftest below:
  1. g = gradient.astype(int): gradients < 1 become 0 (on an image in
     [0,1] then NO pixel votes at all; on the FAST image, in the hundreds,
     the direction is coarsely quantized).
  2. g // g_norm: integer division -- every direction component becomes
     {-1, 0} (1 only when exactly axis-aligned). The votes then only land
     in the "left/top" neighbours, and the peak drifts up to n pixels
     diagonally away from the bleed centre.
  3. O_n and M_n are divided per slice by their maximum instead of being
     capped at k_n: a slice without a bleed gets pulled up to the same
     height as one with a bleed (noise becomes signal), and the scale
     jumps from slice to slice.

Here, per slice (paper notation):
  g      = Sobel gradient in-plane (3x3, /8 -> intensity per voxel),
           image first divided by the case's brain median: |g| is thus
           "contrast relative to tissue per voxel", comparable across cohorts.
  u      = g / |g|  (true division)
  p+     = p + round(n*u)   (bright: gradient points into the bright area, i.e.
                             toward the centre of a bright spot; the gitter_c
                             image is INVERTED, bleeds are bright -> mode "hell")
  O_n(p+) += 1,  M_n(p+) += |g|        (np.bincount = np.add.at, faster)
  Õ_n    = O_n capped at k_n  (k_n = 9.9 for n=1, otherwise 8)
  F_n    = (M_n / k_n) * (|Õ_n| / k_n)^alpha,  alpha = 2
  S_n    = F_n * Gauss(sigma = 0.25 n)
  S      = mean over n
Radii in VOXELS; at 0.5 mm in-plane, [2,3,4,6] = 1-3 mm, the same physical
size in all cohorts (unlike the old [2,3] on the original grid: 0.9-1.4 mm
in cohort1/cohort2, 2-3 mm in cohort3).

Brain edge: outside the brain, the slice is filled with the slice's brain
median before the FRST (otherwise the edge is the strongest boundary in the
image and produces ring responses); the mask used for this is eroded by
1 voxel (the linearly interpolated edge rim does not count as tissue). After
the FRST: 0 outside the brain, divided per case by the 99.9th percentile
within the brain, capped to [0,1] -- CONTINUOUS, no threshold.

  python frst.py --selftest
"""
import sys, time
import numpy as np
from scipy import ndimage

RADIEN = (2, 3, 4, 6)     # voxels at 0.5 mm = 1, 1.5, 2, 3 mm
ALPHA = 2.0
PERZENTIL = 99.9


def k_n(n):
    return 9.9 if n == 1 else 8.0


def gradient_2d(s):
    """Sobel 3x3 in-plane, /8: derivative in intensity per voxel.
    Axis 0 = x, axis 1 = y (nibabel order, like the volumes)."""
    d = np.array([-1.0, 0.0, 1.0]) / 2.0
    w = np.array([1.0, 2.0, 1.0]) / 4.0
    gx = ndimage.correlate1d(ndimage.correlate1d(s, d, axis=0, mode="nearest"), w, axis=1, mode="nearest")
    gy = ndimage.correlate1d(ndimage.correlate1d(s, w, axis=0, mode="nearest"), d, axis=1, mode="nearest")
    return gx, gy


def frst_schicht(s, radien=RADIEN, alpha=ALPHA, beta=0.0, modus="hell"):
    """FRST of a 2D slice, without normalization (S = mean of S_n).
    beta: gradients with |g| <= beta do not vote (paper: beta)."""
    s = np.asarray(s, dtype=np.float64)
    X, Y = s.shape
    gx, gy = gradient_2d(s)
    mag = np.hypot(gx, gy)
    ok = mag > max(beta, 1e-12)
    px, py = np.nonzero(ok)
    g = mag[ok]
    ux, uy = gx[ok] / g, gy[ok] / g           # Einheitsgradient, echte Division
    S = np.zeros((X, Y))
    for n in radien:
        kn = k_n(n)
        O = np.zeros(X * Y)
        M = np.zeros(X * Y)
        dx = np.rint(n * ux).astype(np.int64)
        dy = np.rint(n * uy).astype(np.int64)
        stimmen = []
        if modus in ("hell", "beide"):
            stimmen.append((px + dx, py + dy, +1.0))
        if modus in ("dunkel", "beide"):
            stimmen.append((px - dx, py - dy, -1.0))
        for qx, qy, vz in stimmen:
            drin = (qx >= 0) & (qx < X) & (qy >= 0) & (qy < Y)
            flat = qx[drin] * Y + qy[drin]
            O += vz * np.bincount(flat, minlength=X * Y)
            M += vz * np.bincount(flat, weights=g[drin], minlength=X * Y)
        O = np.clip(O, -kn, kn).reshape(X, Y)
        M = M.reshape(X, Y)
        F = (np.abs(M) / kn) * (np.abs(O) / kn) ** alpha
        S += ndimage.gaussian_filter(F, 0.25 * n)
    return S / len(radien)


def hirnmaske(img):
    """Brain = img > 0, holes filled per slice (like microbleednet)."""
    m = img > 0
    for z in range(m.shape[2]):
        m[:, :, z] = ndimage.binary_fill_holes(m[:, :, z])
    return m


def frst_volumen(img, hirn=None, radien=RADIEN, alpha=ALPHA, beta=0.0, modus="hell",
                 perzentil=PERZENTIL):
    """FRST per axial slice (axis 2) of an (x,y,z) volume.
    Returns: (frst float32 in [0,1], info dict)."""
    img = np.asarray(img, dtype=np.float32)
    if hirn is None:
        hirn = hirnmaske(img)
    innen = np.zeros_like(hirn)
    for z in range(hirn.shape[2]):
        innen[:, :, z] = ndimage.binary_erosion(hirn[:, :, z])
    ref = float(np.median(img[innen])) if innen.any() else 1.0   # tissue level of the case
    out = np.zeros(img.shape, dtype=np.float32)
    for z in range(img.shape[2]):
        if not innen[:, :, z].any():
            continue
        s = img[:, :, z].astype(np.float64) / ref
        med = np.median(s[innen[:, :, z]])
        s[~innen[:, :, z]] = med            # edge must not be a boundary
        out[:, :, z] = frst_schicht(s, radien, alpha, beta, modus)
    out[~hirn] = 0.0
    p = float(np.percentile(out[hirn], perzentil)) if hirn.any() else 0.0
    if p > 0:
        out = np.clip(out / p, 0.0, 1.0)
    return out.astype(np.float32), {"hirn_median_bild": round(ref, 4),
                                     "p999_roh": p, "radien": list(radien),
                                     "alpha": alpha, "beta": beta, "modus": modus}


# ----------------------------------------------------------------- Self-test
def _alt_frst(s, radii=(2, 3), bright=False, dark=False):
    """The old implementation (microbleednet), imported for comparison."""
    sys.path.insert(0, "/home/uchralt/software/microbleed-detection")
    from microbleednet.scripts.data_preparation import fast_radial_symmetry_xfm
    with np.errstate(all="ignore"):
        r = fast_radial_symmetry_xfm(s, np.array(radii), alpha=2, factor_std=0.1,
                                     bright=bright, dark=dark)
    return np.nan_to_num(r)


def _scheibe(X, Y, cx, cy, r):
    xx, yy = np.mgrid[0:X, 0:Y]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(float)


def selftest():
    rng = np.random.default_rng(0)
    X, Y = 200, 200                    # 0.5 mm per voxel -> 100 x 100 mm
    kontrast = 0.3
    img = np.full((X, Y), 0.6)
    # bright disks (inverted image: bleed bright): r = 2 mm (4 vx), r = 4 mm (8 vx)
    scheiben = {"r2mm": (50, 50, 4), "r4mm": (50, 140, 8)}
    for cx, cy, r in scheiben.values():
        img += kontrast * _scheibe(X, Y, cx, cy, r)
    # bright lines (vessel) of equal contrast: width 1 mm (2 vx) and 2 mm (4 vx)
    linien = {"b1mm": (120, 2), "b2mm": (160, 4)}
    for x0, b in linien.values():
        img[x0 - b // 2: x0 - b // 2 + b, 20:180] += kontrast
    img = ndimage.gaussian_filter(img, 0.7)        # partial volume
    img += rng.normal(0, 0.02, img.shape)          # noise, SNR ~ 15

    t = time.time()
    S = frst_schicht(img / np.median(img))
    dt = time.time() - t
    ok = True
    print(f"FRST new, 200x200, radii {RADIEN}: {dt*1000:.0f} ms")
    peak = {}
    for name, (cx, cy, r) in scheiben.items():
        w = 2 * r
        fen = S[cx - w: cx + w + 1, cy - w: cy + w + 1]
        i, j = np.unravel_index(np.argmax(fen), fen.shape)
        off = (int(i - w), int(j - w))
        peak[name] = fen.max()
        tol = 1 if r <= 4 else 2
        good = max(abs(off[0]), abs(off[1])) <= tol
        ok &= good
        print(f"  Disk {name}: peak {fen.max():.4f} at offset {off} vx from the centre "
              f"(tolerance {tol}) {'OK' if good else 'FAIL'}")
    for name, (x0, b) in linien.items():
        lin = S[x0 - 6: x0 + 7, 40:160].max()      # line ends (20/180) excluded
        q = lin / peak["r2mm"]
        good = q < 0.5
        ok &= good
        print(f"  Line {name}: max {lin:.4f} = {q:.2f} x disk r2mm  "
              f"({'OK, clearly lower' if good else 'FAIL'})")
    # Edge test: brain as an ellipse on 0 (only tissue + disk r2mm, no lines),
    # with/without median fill; normalized to the maximum so nothing is capped.
    xx, yy = np.mgrid[0:X, 0:Y]
    ell = ((xx - 100) / 90.0) ** 2 + ((yy - 100) / 70.0) ** 2 <= 1
    hg = ndimage.gaussian_filter(0.6 + kontrast * _scheibe(X, Y, 100, 100, 4), 0.7) \
        + rng.normal(0, 0.02, (X, Y))
    vol = np.where(ell, hg, 0.0)[:, :, None]
    f_mit, _ = frst_volumen(vol, perzentil=100.0)
    roh = frst_schicht(vol[:, :, 0])
    rand = ell & ~ndimage.binary_erosion(ell, iterations=6)
    q_mit = f_mit[:, :, 0][rand].max() / f_mit[:, :, 0][97:104, 97:104].max()
    q_ohne = roh[rand].max() / roh[97:104, 97:104].max()
    ok &= q_mit < 0.5
    print(f"  Brain edge (6-vx rim) / disk r2mm: without median fill {q_ohne:.2f}, "
          f"with {q_mit:.2f} {'OK' if q_mit < 0.5 else 'FAIL'}")

    # Old implementation on the same slice. In preprocess_subject it ran with
    # dark=True on the NON-inverted FAST image (intensities in the hundreds):
    # there bleeds are dark -> same scene inverted and x1000.
    alt_dunkel = _alt_frst((1.0 - img) * 1000.0, dark=True)
    alt_hell = _alt_frst(img * 1000.0, bright=True)
    alt_01 = _alt_frst(img, bright=True)
    for lab, A in (("old dark=True, 1000*(1-img) [like preprocess]", alt_dunkel),
                   ("old bright=True, 1000*img", alt_hell)):
        for name, (cx, cy, r) in scheiben.items():
            w = 2 * r
            fen = A[cx - w: cx + w + 1, cy - w: cy + w + 1]
            i, j = np.unravel_index(np.argmax(fen), fen.shape)
            print(f"  {lab}: disk {name} peak at offset ({int(i - w)}, {int(j - w)}) vx")
    print(f"  old bright=True on image in [0,1]: max {alt_01.max():.3g} "
          f"(int cast: gradients < 1 -> 0, no vote)")
    # Direction skew directly: distribution of vote directions old vs new
    gx, gy = np.gradient(img * 1000.0)
    g = np.stack([gx, gy], -1).astype(int)
    gn = np.sqrt((g ** 2).sum(-1))
    m = gn > 0
    with np.errstate(all="ignore"):
        q = np.floor_divide(g[m], gn[m][:, None])
    werte = np.unique(q)
    print(f"  old directions g//|g| only take the values {werte.tolist()}; "
          f"share of votes with component +1: {(q == 1).any(-1).mean():.4f}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
