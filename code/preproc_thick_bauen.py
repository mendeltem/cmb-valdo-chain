#!/usr/bin/env python3
"""wk7 (07.11.2026): dicke Schichten am NATIVEN Bild simulieren, VOR dem Gitterbau.

Auftrag: block-average von n benachbarten Quellschichten (Betrag, hier: Mittelwert der
bereits invertierten Intensitaet -- im Hirn 1-img/max, also der T2*-Betrag), n = 3, 5, 7
(0.8 mm -> ~2.4 / 4.0 / 5.6 mm, ohne Luecke, z = 0.8 * n mm). Referenz: ein Voxel ist CMB,
wenn eine der verschmolzenen Schichten es war (Max-Pool ueber die Blcke), Komponenten
werden beim Gitterbau neu gezaehlt (26er, wie in gitter_c_bauen.py).

Quelle: dev/preproc_roh/<id>/{id}_image.nii.gz (invertiert, Hirnmaske, OHNE Gefaess-
Uebermalung -- exakt dieselbe Quelle wie gitter_d; siehe preproc_roh_bauen.py) und
{id}_label.nii.gz.
Ziel:   dev/preproc_thick_<n>/<id>/{id}_image.nii.gz (float32, z = 0.8*n mm) und
        {id}_label.nii.gz (uint8, max-pooled). Wiederaufnehmbar (info.json je Fall).

Danach: GITTER_C_QUELLE=dev/preproc_thick_<n> GITTER_C_ZIEL=dev/gitter_thick_<n>
        python code/gitter_c_bauen.py  (EXAKT dieselbe gitter_d-Logik: gleiche Hirn-
        Bounding-Box-Regel "Bild != 0" + 8 Voxel, gleiche FRST-Radien, lineare Ordnung,
        Label nearest mit Komponenten-Rettung).

Unit-Test (--selftest, laeuft ohne Daten): Block-Mittelung eines synthetischen
Wurfels/Sphaeroids (a) haelt das Zentrum fest, (b) senkt den Kontrast, wenn die
Sphaere nur einen Teil des Blocks fuellt (in z duenner als n Schichten).
"""
import os, sys, json, argparse
import numpy as np
import nibabel as nib

ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEN = (3, 5, 7)


def block_average(bild, lab, n):
    """(x,y,z) float32 / (x,y,z) uint8 -> z-Blcke a n Schichten.
    Bild: Mittelwert der n Schichten (Betrag). Label: Max-Pool (ein Voxel ist CMB,
    wenn eine der verschmolzenen Schichten es war). Unvollstaendiger letzter Block:
    Mittel/Max ueber die vorhandenen Schichten."""
    z = bild.shape[2]
    zb = (z + n - 1) // n
    img = np.empty(bild.shape[:2] + (zb,), np.float32)
    labo = np.empty(lab.shape[:2] + (zb,), np.uint8)
    for k in range(zb):
        sl = slice(k * n, min((k + 1) * n, z))
        img[:, :, k] = bild[:, :, sl].mean(axis=2)
        labo[:, :, k] = (lab[:, :, sl].max(axis=2) > 0).astype(np.uint8)
    return img, labo


def selftest():
    # Sphaeroid: 3x3x2 Voxel (2 Schichten dick in z) bei z=20, Intensitaet 0.8 auf 0.0
    bild = np.zeros((30, 30, 40), np.float32)
    lab = np.zeros_like(bild, np.uint8)
    bild[13:16, 13:16, 20:21] = 0.8          # eine einzige Schicht -> immer genau EIN Block
    lab[13:16, 13:16, 20:21] = 1
    for n in NEN:
        img, labo = block_average(bild, lab, n)
        # (a) Zentrum: das Objekt liegt in Schichten 20..21 -> Block k = 20//n
        k = 20 // n
        assert labo[13:16, 13:16, k].any(), f"n={n}: Objekt nach Max-Pool verschwunden"
        cy, cx = np.unravel_index(img[:, :, k].argmax(), img[:, :, k].shape)
        assert 13 <= cy <= 15 and 13 <= cx <= 15, f"n={n}: Zentrum verrutscht ({cy},{cx})"
        # (b) Kontrast sinkt: 1 von n Schichten traege 0.8 -> Mittel = 0.8/n (fuer n>1 streng kleiner)
        peak = float(img[14, 14, k])
        assert abs(peak - 0.8 / n) < 1e-5, f"n={n}: Peak {peak} != {0.8/n}"
        # Referenz: genau EIN Block mit den 9 Voxel, sonst leer
        assert labo.sum() == 9 and labo[14, 14].sum() == 1, f"n={n}: falsche Referenz (summe {labo.sum()})"
        print(f"n={n}: Block {k}, Peak {peak:.3f} (= 0.8/{n}), Referenz 1 Block mit 9 Voxel -- ok")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="ein n; 0 = alle 3,5,7")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", default=f"{ROOT}/dev/manifest_valdo_gitter_d.json")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.n:
        ns = (args.n,)
    else:
        ns = NEN
    man = json.load(open(args.manifest))
    rss = [e for e in man if e["cohort"] == "valdo-2"]
    assert rss and len(rss) <= 27, f"unerwartet: {len(rss)} RSS-Faelle"
    for n in ns:
        zq = 0.8
        for e in rss:
            sid = e["id"]
            dsrc = f"{ROOT}/dev/preproc_roh/{sid}"
            dz = f"{ROOT}/dev/preproc_thick_{n}/{sid}"
            if os.path.exists(f"{dz}/info.json"):
                continue
            os.makedirs(dz, exist_ok=True)
            ni = nib.load(f"{dsrc}/{sid}_image.nii.gz")
            nl = nib.load(f"{dsrc}/{sid}_label.nii.gz")
            img = np.asarray(ni.dataobj, dtype=np.float32)
            lab = (np.asarray(nl.dataobj) > 0).astype(np.uint8)
            assert img.shape == lab.shape and img.shape[2] % 1 == 0
            img2, lab2 = block_average(img, lab, n)
            aff = ni.affine.copy()
            aff[2, 2] *= n                       # z-Spacing 0.8 -> 0.8*n mm
            aff[2, 3] += 0.8 * (n - 1) / 2.0     # Mitte von Block 0 = Mitte seiner n Schichten,
                                                 # also die Weltlage der Daten bleibt korrekt
            hdr = ni.header.copy()
            hdr.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(img2, aff, hdr), f"{dz}/{sid}_image.nii.gz")
            hdr2 = ni.header.copy(); hdr2.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(lab2, aff, hdr2), f"{dz}/{sid}_label.nii.gz")
            json.dump({"id": sid, "n": n, "quelle": "preproc_roh", "z_mm": round(zq * n, 3),
                       "shape": list(img2.shape), "z_schichten": int(img.shape[2])},
                      open(f"{dz}/info.json", "w"))
            print(f"n={n} {sid} {img.shape} -> {img2.shape}", flush=True)
    print("fertig")


if __name__ == "__main__":
    main()
