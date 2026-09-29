#!/usr/bin/env python3
"""wb2 Schritt 1: 5-Fach-CV-Falten aus eingaben/swi/preproc2 bauen (Symlinks).

Je Falte k: eingaben/swi/falten/fold{k}/{images,labels,frsts}/
  images:  alle 61 Faelle (Train+Test zusammen -- fine_tune trennt mit --test_ids)
  labels:  alle 61 Faelle
  frsts:   alle 61 Faelle
  test_ids.txt: die Testfaelle der Falte (eine ID pro Zeile)
Summe der Testfaelle ueber die Falten muss 61 sein (jeder Fall genau einmal).
"""
import json, os

B = os.path.dirname(os.path.abspath(__file__))
faelle = json.load(open(f"{B}/faelle.json"))["swi"]["faelle"]
SRC = f"{B}/eingaben/swi/preproc2"
DST = f"{B}/eingaben/swi/falten"

os.makedirs(DST, exist_ok=True)
gesamt_test = 0
for k in range(5):
    fold = f"{DST}/fold{k}"
    test_ids = [c["id"] for c in faelle if c["fold"] == k]
    for sub in ("images", "labels", "frsts"):
        d = f"{fold}/{sub}"
        os.makedirs(d, exist_ok=True)
        for c in faelle:
            cid = c["id"]
            if sub == "images":
                src = f"{SRC}/images/{cid}_preproc.nii.gz"
            elif sub == "labels":
                src = f"{SRC}/labels/{cid}_label.nii.gz"
            else:
                src = f"{SRC}/frsts/{cid}_frst.nii.gz"
            dst = f"{d}/{os.path.basename(src)}"
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            if not os.path.exists(src):
                raise SystemExit(f"Quelle fehlt: {src}")
            os.symlink(os.path.abspath(src), dst)
    with open(f"{fold}/test_ids.txt", "w") as f:
        f.write("\n".join(test_ids) + "\n")
    n_img = len(os.listdir(f"{fold}/images"))
    n_lab = len(os.listdir(f"{fold}/labels"))
    n_frs = len(os.listdir(f"{fold}/frsts"))
    print(f"fold{k}: {len(test_ids)} Testfaelle, {n_img} images, {n_lab} labels, {n_frs} frsts")
    if not (n_img == n_lab == n_frs == 61):
        raise SystemExit(f"fold{k}: Erwartet 61/61/61, ist {n_img}/{n_lab}/{n_frs}")
    gesamt_test += len(test_ids)

print(f"Summe Testfaelle ueber alle Falten: {gesamt_test}")
if gesamt_test != 61:
    raise SystemExit(f"Summe Testfaelle {gesamt_test} != 61")
print("OK: 5 Falten gebaut.")
