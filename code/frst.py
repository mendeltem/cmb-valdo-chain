#!/usr/bin/env python3
"""FRST-Kanal neu (Claude 11.09.2026): Fast Radial Symmetry Transform nach
Loy & Zelinsky (2003, IEEE PAMI 25(8)), je axialer Schicht, vektorisiert.

Warum neu: die alte Umsetzung (microbleednet, scripts/data_preparation.py
fast_radial_symmetry_xfm, ~Z. 530-590) hat drei Fehler, die das Signal
verzerren -- gemessen im --selftest unten:
  1. g = gradient.astype(int): Gradienten < 1 werden 0 (auf einem Bild in
     [0,1] stimmt dann KEIN Pixel ab; auf dem FAST-Bild in Hunderten wird
     die Richtung grob quantisiert).
  2. g // g_norm: Ganzzahl-Division -- jede Richtungskomponente wird zu
     {-1, 0} (1 nur exakt achsparallel). Die Stimmen landen damit nur in
     den Nachbarn "links/oben", der Gipfel wandert um bis zu n Pixel
     diagonal von der Blutungsmitte weg.
  3. O_n und M_n werden je Schicht durch ihr Maximum geteilt statt an k_n
     gekappt: eine Schicht ohne Blutung wird auf dieselbe Hoehe gezogen wie
     eine mit (Rauschen wird Signal), und die Skala springt von Schicht zu
     Schicht.

Hier, je Schicht (Paper-Notation):
  g      = Sobel-Gradient in der Ebene (3x3, /8 -> Intensitaet je Voxel),
           Bild vorher durch den Hirn-Median des Falls geteilt: |g| ist damit
           "Kontrast relativ zum Gewebe je Voxel", ueber Kohorten vergleichbar.
  u      = g / |g|  (echte Division)
  p+     = p + round(n*u)   (hell: Gradient zeigt ins Helle, also zur Mitte
                             eines hellen Flecks; das gitter_c-Bild ist
                             INVERTIERT, Blutungen sind hell -> Modus "hell")
  O_n(p+) += 1,  M_n(p+) += |g|        (np.bincount = np.add.at, schneller)
  Õ_n    = O_n gekappt bei k_n  (k_n = 9.9 fuer n=1, sonst 8)
  F_n    = (M_n / k_n) * (|Õ_n| / k_n)^alpha,  alpha = 2
  S_n    = F_n * Gauss(sigma = 0.25 n)
  S      = Mittel ueber n
Radien in VOXELN; auf 0.5 mm in der Ebene sind [2,3,4,6] = 1-3 mm, in allen
Kohorten dieselbe physische Groesse (anders als die alten [2,3] auf dem
Originalgitter: 0.9-1.4 mm in Ko1/Ko2, 2-3 mm in Ko3).

Hirnrand: ausserhalb des Hirns wird vor der FRST mit dem Hirn-Median der
Schicht gefuellt (sonst ist der Rand die staerkste Kante im Bild und macht
Ringantworten), die Maske dafuer um 1 Voxel erodiert (der linear
interpolierte Randsaum zaehlt nicht als Gewebe). Nach der FRST: ausserhalb
des Hirns 0, je Fall durch das 99.9-Perzentil im Hirn geteilt, auf [0,1]
gekappt -- KONTINUIERLICH, keine Schwelle.

  python frst.py --selftest
"""
import sys, time
import numpy as np
from scipy import ndimage

RADIEN = (2, 3, 4, 6)     # Voxel auf 0.5 mm = 1, 1.5, 2, 3 mm
ALPHA = 2.0
PERZENTIL = 99.9


def k_n(n):
    return 9.9 if n == 1 else 8.0


def gradient_2d(s):
    """Sobel 3x3 in der Ebene, /8: Ableitung in Intensitaet je Voxel.
    Achse 0 = x, Achse 1 = y (nibabel-Reihenfolge, wie die Volumen)."""
    d = np.array([-1.0, 0.0, 1.0]) / 2.0
    w = np.array([1.0, 2.0, 1.0]) / 4.0
    gx = ndimage.correlate1d(ndimage.correlate1d(s, d, axis=0, mode="nearest"), w, axis=1, mode="nearest")
    gy = ndimage.correlate1d(ndimage.correlate1d(s, w, axis=0, mode="nearest"), d, axis=1, mode="nearest")
    return gx, gy


def frst_schicht(s, radien=RADIEN, alpha=ALPHA, beta=0.0, modus="hell"):
    """FRST einer 2D-Schicht, ohne Normierung (S = Mittel der S_n).
    beta: Gradienten mit |g| <= beta stimmen nicht ab (Paper: beta)."""
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
    """Hirn = img > 0, Loecher je Schicht gefuellt (wie microbleednet)."""
    m = img > 0
    for z in range(m.shape[2]):
        m[:, :, z] = ndimage.binary_fill_holes(m[:, :, z])
    return m


def frst_volumen(img, hirn=None, radien=RADIEN, alpha=ALPHA, beta=0.0, modus="hell",
                 perzentil=PERZENTIL):
    """FRST je axialer Schicht (Achse 2) eines (x,y,z)-Volumens.
    Rueckgabe: (frst float32 in [0,1], info-dict)."""
    img = np.asarray(img, dtype=np.float32)
    if hirn is None:
        hirn = hirnmaske(img)
    innen = np.zeros_like(hirn)
    for z in range(hirn.shape[2]):
        innen[:, :, z] = ndimage.binary_erosion(hirn[:, :, z])
    ref = float(np.median(img[innen])) if innen.any() else 1.0   # Gewebeniveau des Falls
    out = np.zeros(img.shape, dtype=np.float32)
    for z in range(img.shape[2]):
        if not innen[:, :, z].any():
            continue
        s = img[:, :, z].astype(np.float64) / ref
        med = np.median(s[innen[:, :, z]])
        s[~innen[:, :, z]] = med            # Rand darf keine Kante sein
        out[:, :, z] = frst_schicht(s, radien, alpha, beta, modus)
    out[~hirn] = 0.0
    p = float(np.percentile(out[hirn], perzentil)) if hirn.any() else 0.0
    if p > 0:
        out = np.clip(out / p, 0.0, 1.0)
    return out.astype(np.float32), {"hirn_median_bild": round(ref, 4),
                                     "p999_roh": p, "radien": list(radien),
                                     "alpha": alpha, "beta": beta, "modus": modus}


# ----------------------------------------------------------------- Selbsttest
def _alt_frst(s, radii=(2, 3), bright=False, dark=False):
    """Die alte Umsetzung (microbleednet), zum Vergleich importiert."""
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
    X, Y = 200, 200                    # 0.5 mm je Voxel -> 100 x 100 mm
    kontrast = 0.3
    img = np.full((X, Y), 0.6)
    # helle Scheiben (invertiertes Bild: Blutung hell): r = 2 mm (4 vx), r = 4 mm (8 vx)
    scheiben = {"r2mm": (50, 50, 4), "r4mm": (50, 140, 8)}
    for cx, cy, r in scheiben.values():
        img += kontrast * _scheibe(X, Y, cx, cy, r)
    # helle Linien (Gefaess) gleichen Kontrasts: Breite 1 mm (2 vx) und 2 mm (4 vx)
    linien = {"b1mm": (120, 2), "b2mm": (160, 4)}
    for x0, b in linien.values():
        img[x0 - b // 2: x0 - b // 2 + b, 20:180] += kontrast
    img = ndimage.gaussian_filter(img, 0.7)        # Partialvolumen
    img += rng.normal(0, 0.02, img.shape)          # Rauschen, SNR ~ 15

    t = time.time()
    S = frst_schicht(img / np.median(img))
    dt = time.time() - t
    ok = True
    print(f"FRST neu, 200x200, Radien {RADIEN}: {dt*1000:.0f} ms")
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
        print(f"  Scheibe {name}: Gipfel {fen.max():.4f} bei Versatz {off} vx von der Mitte "
              f"(Toleranz {tol}) {'OK' if good else 'FEHLER'}")
    for name, (x0, b) in linien.items():
        lin = S[x0 - 6: x0 + 7, 40:160].max()      # Linienenden (20/180) ausgespart
        q = lin / peak["r2mm"]
        good = q < 0.5
        ok &= good
        print(f"  Linie {name}: max {lin:.4f} = {q:.2f} x Scheibe r2mm  "
              f"({'OK, klar niedriger' if good else 'FEHLER'})")
    # Rand-Test: Hirn als Ellipse auf 0 (nur Gewebe + Scheibe r2mm, keine Linien),
    # mit/ohne Median-Fuellung; Normierung aufs Maximum, damit nichts kappt.
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
    print(f"  Hirnrand (6-vx-Saum) / Scheibe r2mm: ohne Median-Fuellung {q_ohne:.2f}, "
          f"mit {q_mit:.2f} {'OK' if q_mit < 0.5 else 'FEHLER'}")

    # Alte Umsetzung auf derselben Schicht. Sie lief in preprocess_subject mit
    # dark=True auf dem NICHT invertierten FAST-Bild (Intensitaeten in Hunderten):
    # dort sind Blutungen dunkel -> dieselbe Szene invertiert und x1000.
    alt_dunkel = _alt_frst((1.0 - img) * 1000.0, dark=True)
    alt_hell = _alt_frst(img * 1000.0, bright=True)
    alt_01 = _alt_frst(img, bright=True)
    for lab, A in (("alt dark=True, 1000*(1-img) [wie preprocess]", alt_dunkel),
                   ("alt bright=True, 1000*img", alt_hell)):
        for name, (cx, cy, r) in scheiben.items():
            w = 2 * r
            fen = A[cx - w: cx + w + 1, cy - w: cy + w + 1]
            i, j = np.unravel_index(np.argmax(fen), fen.shape)
            print(f"  {lab}: Scheibe {name} Gipfel bei Versatz ({int(i - w)}, {int(j - w)}) vx")
    print(f"  alt bright=True auf Bild in [0,1]: max {alt_01.max():.3g} "
          f"(int-Cast: Gradienten < 1 -> 0, keine Stimme)")
    # Richtungsschiefe direkt: Verteilung der Stimmrichtungen alt vs neu
    gx, gy = np.gradient(img * 1000.0)
    g = np.stack([gx, gy], -1).astype(int)
    gn = np.sqrt((g ** 2).sum(-1))
    m = gn > 0
    with np.errstate(all="ignore"):
        q = np.floor_divide(g[m], gn[m][:, None])
    werte = np.unique(q)
    print(f"  alte Richtungen g//|g| nehmen nur die Werte {werte.tolist()} an; "
          f"Anteil Stimmen mit Komponente +1: {(q == 1).any(-1).mean():.4f}")
    print("SELFTEST", "BESTANDEN" if ok else "FEHLGESCHLAGEN")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
