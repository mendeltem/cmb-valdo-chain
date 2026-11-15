#!/usr/bin/env python3
"""we2 Gitter: 3D-Attention-U-Net trainieren, EINE Variante, Falte 0 als Test.

  gitter_train.py <nativ|gemeinsam> [zeitlimit_min] [--limit N]

Eingaben:
  nativ:      dev/preproc/<id>/  (Originalgitter je Fall),  Patch 48x64x96 (z,y,x)
  gemeinsam:  dev/gitter_b/<id>/ (0.5 x 0.5 x 1.0 mm),      Patch 48x64x96 (z,y,x)
Beide: 2 Kanaele T2* + FRST, lr 3e-4, Loss 0.5 BCE + 0.5 Dice,
gleiche Epochenzahl/Hyperparameter (konstant unten), Seed 42.
Falte 0 = Testfalte (kein Training, keine Val), Rest (Falten 1-4) Training.
Ein Fall wird genau EINMAL vorhergesagt: auf dem Gitter seiner Eingabevariante.

Achsenkonvention durchgehend (z, y, x) = NIfTI (D, H, W).

Ausgabe: ergebnisse/gitter-<variante>/
  modell.pt, karten/<id>_pred.nii.gz + _prob.nii.gz (Eingabegitter der Variante),
  train.log, vram.log, lauf.json
"""
import os, sys, json, time, random, threading, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

DEV = "cuda"
ROOT = os.environ.get("CMB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Patch (z, y, x), NIfTI-Achsenkonvention:
# A nativ:    96x64x48 als x*y*z = (48, 64, 96) z*y*x -- wie das SWI-Netz aus wd3
# B gemeinsam: A in Ko2 (46.9 x 31.3 x 38.4 mm) / 0.5x0.5x1.0 mm
#              = 94x62x38 als x*y*z = (38, 62, 94) z*y*x  -> 98.4 % von As Volumen
PATCHES = {"nativ": (48, 64, 96), "gemeinsam": (38, 62, 94)}
PATCH = None  # wird in main() je Variante gesetzt
BS = 4
LR = 3e-4
ES = 10
MIN_EPOCHS = 15
MAX_EPOCHS = 40
N_PATCHES = 800        # Proben je Epoche (60/25/15)
SEED = 42


def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def log(msg, fh):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def dice_bce(logits, ziel):
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, ziel)
    pf, zf = p.flatten(1), ziel.flatten(1)
    inter = (pf * zf).sum(1)
    dice = 1 - (2 * inter + 1.0) / (pf.sum(1) + zf.sum(1) + 1.0)
    return 0.5 * bce + 0.5 * dice.mean()


class Daten:
    """Ladt Trainings- (Falten 1-4) und Testfaelle (Falte 0) in den Speicher.

    Kanal 0 = T2*-Bild, Kanal 1 = FRST (binarisiert >0.5). Bei 'nativ' bleibt
    jeder Fall auf seinem Originalgitter, bei 'gemeinsam' liegen alle auf
    0.5x0.5x1.0 mm. Achsen (z, y, x).
    """

    def __init__(self, variant):
        src = "dev/preproc" if variant == "nativ" else "dev/gitter_b"
        cases = json.load(open(f"{ROOT}/dev/cases.json"))["t2star"]["cases"]
        self.train = [self._one(c, src) for c in cases if 0 < c["fold"] <= 4]
        self.test = [self._one(c, src) for c in cases if c["fold"] == 0]
        self.pos = [np.argwhere(l > 0.5) for _, _, _, l, _, _ in self.train]
        self.neg = [np.argwhere(b & (l <= 0.5)) for _, _, _, l, b, _ in self.train]
        if not any(p.shape[0] for p in self.pos):
            raise SystemExit("keine Positiv-Voxel im Training")

    def _one(self, c, src):
        d = f"{ROOT}/{src}/{c['id']}"
        img = np.asanyarray(nib.load(f"{d}/{c['id']}_image.nii.gz").dataobj, dtype=np.float32)
        frs = np.asanyarray(nib.load(f"{d}/{c['id']}_frst.nii.gz").dataobj, dtype=np.float32) > 0.5
        lab = np.asanyarray(nib.load(f"{d}/{c['id']}_label.nii.gz").dataobj, dtype=np.float32)
        orig = img.shape
        # Achsen kleiner als den Patch (Ko1 z=35 < 48) werden 0-gespeichert,
        # damit jeder Patch 48x64x96 ist; Output wird zurueckgeschnitten.
        pad = [(0, max(PATCH[0] - orig[0], 0)), (0, max(PATCH[1] - orig[1], 0)),
               (0, max(PATCH[2] - orig[2], 0))]
        if any(p[1] for p in pad):
            img = np.pad(img, pad, mode="constant")
            frs = np.pad(frs, pad, mode="constant")
            lab = np.pad(lab, pad, mode="constant")
        brain = (img > 0)
        return (c["id"], img, frs, lab, brain, orig)

    def pos_center(self, rng):
        for _ in range(30):
            i = rng.randint(len(self.train))
            if self.pos[i].shape[0]:
                return i, tuple(self.pos[i][rng.randint(self.pos[i].shape[0])])
        i = rng.randint(len(self.train))
        return i, tuple(np.array(self.train[i][1].shape) // 2)

    def brain_point(self, rng):
        for _ in range(30):
            i = rng.randint(len(self.train))
            if self.neg[i].shape[0]:
                return i, tuple(self.neg[i][rng.randint(self.neg[i].shape[0])])
        i = rng.randint(len(self.train))
        return i, tuple(np.array(self.train[i][1].shape) // 2)


def center_to_start(c, shape):
    return tuple(int(min(max(c[i] - PATCH[i] // 2, 0), shape[i] - PATCH[i])) for i in range(3))


def batch(fa, idx, centers):
    xs, ys = [], []
    for i, c in zip(idx, centers):
        _, img, frs, lab, _, _ = fa.train[i]
        z, y, x = center_to_start(c, img.shape)
        xs.append(np.stack([img[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]],
                            frs[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]].astype(np.float32)], 0))
        ys.append(lab[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]])
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)).unsqueeze(1))


def validate_pos(model, fa, rng, n=120):
    model.eval()
    tot_p = tot_y = inter = 0
    with torch.no_grad():
        for i in range(n):
            if i % 2 == 0:
                ci, c = fa.pos_center(rng)
            else:
                ci, c = fa.brain_point(rng)
            _, img, frs, lab, _, orig = fa.train[ci]
            z, y, x = center_to_start(c, img.shape)
            p = np.stack([img[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]],
                          frs[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]].astype(np.float32)], 0)
            l = lab[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] > 0.5
            out = torch.sigmoid(model(torch.from_numpy(p[None]).to(DEV))).cpu().numpy()[0, 0]
            pb = out > 0.5
            tot_p += pb.sum(); tot_y += l.sum(); inter += np.logical_and(pb, l).sum()
    return 2 * inter / max(tot_p + tot_y, 1)


def prob_map(model, img, frs):
    """Ganzbild-Wahrscheinlichkeit auf dem (Eingabe-)Gitter, Tiling 48x64x96 (z,y,x),
    Stride halbe Patchgroesse, weiche Aufsummung mit Normalisierung.
    Kleine z-Achse wird 0-gespeichert, damit der Patch immer passt;
    die Ausgabe wird zurueckgeschnitten."""
    D, H, W = img.shape
    pad = [(0, max(PATCH[0] - D, 0)), (0, max(PATCH[1] - H, 0)), (0, max(PATCH[2] - W, 0))]
    imgp = np.pad(img, pad, mode="constant")
    frsp = np.pad(frs, pad, mode="constant")
    Dp, Hp, Wp = imgp.shape
    sz, sy, sx = Dp - PATCH[0] + 1, Hp - PATCH[1] + 1, Wp - PATCH[2] + 1
    zs = list(range(0, sz, PATCH[0] // 2))
    ys = list(range(0, sy, PATCH[1] // 2))
    xs = list(range(0, sx, PATCH[2] // 2))
    for lst, sz, P in ((zs, sz, PATCH[0]), (ys, sy, PATCH[1]), (xs, sx, PATCH[2])):
        if lst[-1] + P < sz + P - 1:  # letzter Rand nicht abgedeckt
            lst.append(sz - 1)
    acc = np.zeros((Dp, Hp, Wp), np.float64)
    wgt = np.zeros((Dp, Hp, Wp), np.float64)
    model.eval()
    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    p = np.stack([imgp[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]],
                                  frsp[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]].astype(np.float32)], 0)
                    out = torch.sigmoid(model(torch.from_numpy(p[None]).to(DEV))).cpu().numpy()[0, 0]
                    acc[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += out
                    wgt[z:z + PATCH[0], y:y + PATCH[1], x:x + PATCH[2]] += 1.0
    return (acc / wgt)[:D, :H, :W].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variante", choices=["nativ", "gemeinsam"])
    ap.add_argument("zeitlimit", type=float, nargs="?", default=90.0)  # Minuten
    ap.add_argument("--limit", type=int, default=None, help="max. Trainingsfaelle (Probelauf)")
    ap.add_argument("--ep", type=int, default=None, help="feste Epochenzahl (ueberschreibt MAX_EPOCHS, ES aus)")
    a = ap.parse_args()
    global PATCH
    PATCH = PATCHES[a.variante]
    seed_all(SEED)

    z = f"{ROOT}/ergebnisse/gitter-{a.variante}"
    os.makedirs(f"{z}/karten", exist_ok=True)
    fh = open(f"{z}/train.log", "a")
    fa = Daten(a.variante)
    if a.limit is not None:
        fa.train = fa.train[:a.limit]
        fa.pos = fa.pos[:a.limit]
        fa.neg = fa.neg[:a.limit]
    n_pos = int(sum(p.shape[0] for p in fa.pos))
    log(f"we2 Gitter {a.variante}: train {len(fa.train)} Faelle ({n_pos} Positiv-Voxel), "
        f"test {len(fa.test)} Faelle (Falte 0), Patch (z,y,x) {PATCH}", fh)

    # VRAM-Wächter (20 s Takt)
    stop_vram = threading.Event()
    vlog = open(f"{z}/vram.log", "a")
    def vram_watch():
        while not stop_vram.is_set():
            try:
                vlog.write(os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").read().strip() + "\n")
                vlog.flush()
            except Exception:
                pass
            stop_vram.wait(20)
    threading.Thread(target=vram_watch, daemon=True).start()

    sys.path.insert(0, f"{ROOT}/code")
    from attention_unet import AttentionUNet3D  # 2-Kanal-Fassung (T2*+FRST), Kopie in code/
    model = AttentionUNet3D(basis=32).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Params {n_params/1e6:.1f}M, 2 Kanaele (T2*+FRST), {N_PATCHES}/Epoch, "
        f"lr {LR}, ES {ES} ab ep {MIN_EPOCHS}, max {MAX_EPOCHS} ep", fh)

    # Rauchtest: 1 Batch
    ci, c = fa.pos_center(np.random.RandomState(SEED))
    xb, yb = batch(fa, [ci], [c])
    xb, yb = xb.to(DEV), yb.to(DEV)
    t0b = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = dice_bce(model(xb), yb)
    log(f"Rauchtest Batch {tuple(xb.shape)}: loss {loss.item():.4f} in {time.time()-t0b:.1f}s", fh)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_ep = a.ep if a.ep is not None else MAX_EPOCHS
    if a.ep is not None:
        log(f"feste Epochenzahl {n_ep} (--ep), Early-Stop deaktiviert", fh)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    scaler = torch.amp.GradScaler("cuda")
    best_val, best_state, bad = -1.0, None, 0
    epochs_done, t0 = 0, time.time()
    rng = np.random.RandomState(SEED + 7)
    for ep in range(1, n_ep + 1):
        model.train()
        tot = 0.0
        n_pos_ = int(N_PATCHES * 0.6); n_brain = int(N_PATCHES * 0.25)
        idx, centers = [], []
        for _ in range(n_pos_):
            i, c = fa.pos_center(rng); idx.append(i); centers.append(c)
        for _ in range(n_brain):
            i, c = fa.brain_point(rng); idx.append(i); centers.append(c)
        for _ in range(N_PATCHES - n_pos_ - n_brain):
            i = rng.randint(len(fa.train))
            shp = fa.train[i][1].shape
            idx.append(i); centers.append((rng.randint(shp[0]), rng.randint(shp[1]), rng.randint(shp[2])))
        perm = rng.permutation(len(idx))
        for b0 in range(0, len(idx), BS):
            sel = perm[b0:b0 + BS]
            xb, yb = batch(fa, [idx[i] for i in sel], [centers[i] for i in sel])
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = dice_bce(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(sel)
        sched.step()
        val_dice = validate_pos(model, fa, np.random.RandomState(SEED + 100))
        epochs_done = ep
        log(f"ep {ep}: loss {tot/N_PATCHES:.4f} | val_dice {val_dice:.4f} | {time.time()-t0:.0f}s", fh)
        if val_dice > best_val:
            best_val, bad = val_dice, 0
            best_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ES and ep >= MIN_EPOCHS and a.ep is None:
                log(f"Early-Stop ep {ep} (best val {best_val:.4f})", fh)
                break
        if time.time() - t0 > a.zeitlimit * 60:
            log(f"Zeitlimit {a.zeitlimit:.0f} min erreicht (ep {ep})", fh)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{z}/modell.pt")

    # Testkarten: Falte 0, Wahrscheinlichkeit + binarisiert, auf dem Gitter der Variante
    for sid, img, frs, lab, _, orig in fa.test:
        pm = prob_map(model, img, frs)
        pm = pm[:orig[0], :orig[1], :orig[2]]  # 0-Padding abschneiden
        aff = nib.load(f"{ROOT}/dev/gitter_b/{sid}/{sid}_image.nii.gz"
                       if a.variante == "gemeinsam"
                       else f"{ROOT}/dev/preproc/{sid}/{sid}_image.nii.gz").affine
        nib.save(nib.Nifti1Image(pm, aff), f"{z}/karten/{sid}_prob.nii.gz")
        pred = (pm > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(pred, aff), f"{z}/karten/{sid}_pred.nii.gz")
        log(f"karte {sid}: max {pm.max():.3f} | p99.9 {np.percentile(pm,99.9):.3f} | pred-Voxel {int(pred.sum())}", fh)

    stop_vram.set()
    vlog.close()
    vram = [int(x) for x in open(f"{z}/vram.log").read().split() if x.isdigit()]
    json.dump({
        "netz": "attention-unet-3d", "variante": a.variante, "fold": 0,
        "dauer_s": round(time.time() - t0, 1), "epochen": epochs_done,
        "val_dice_best": round(float(best_val), 4),
        "vram_spitze_mib": max(vram) if vram else None,
        "n_karten": len(os.listdir(f"{z}/karten")) // 2,
        "parameter": {"patch": list(PATCH), "patch_konvention": "z_y_x",
                      "bs": BS, "lr": LR, "es": ES,
                      "min_ep": MIN_EPOCHS, "max_ep": MAX_EPOCHS,
                      "n_patches": N_PATCHES, "basis": 32, "kanaele": 2,
                      "loss": "0.5*BCE + 0.5*Dice", "seed": SEED},
    }, open(f"{z}/lauf.json", "w"), indent=1)
    log(f"FERTIG {a.variante}: {epochs_done} ep, val_dice {best_val:.4f}, {time.time()-t0:.0f}s", fh)
    fh.close()


if __name__ == "__main__":
    main()
