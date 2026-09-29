"""Stage 2, step 2: decide for every candidate whether it is a microbleed.

Two classifiers are compared, because with roughly 200 candidates nobody can know in advance
whether a neural network or a handful of hand-made features generalises better:

* ``features`` -- ten interpretable numbers per candidate (stage-1 probability, size, radial
  symmetry, contrast to the surroundings, elongation, distance to the brain edge, ...) fed into a
  regularised logistic regression and into a random forest;
* ``cnn``      -- a small plain 3D CNN (three double-convolution blocks 16-32-64, NO residual connections, 0.22 M
  parameters; earlier notes call it "3D ResNet", which it is not) on a 32 x 32 x 16 voxel patch with three channels
  (T2*, radial symmetry, stage-1 probability), trained with flips and random offsets, three seeds
  averaged.

Evaluation is nested so that no number below is optimistic:
  outer loop = the five folds of the project. The classifier for fold k never sees a candidate of
  fold k. Its acceptance threshold is chosen on out-of-fold scores INSIDE the four training folds
  (inner 4-fold loop) and then applied, unchanged, to fold k.
Scores use the project metric (26-connectivity, touch rule) and are pooled over all 57 cases;
reference lesions that no candidate reaches count as missed, whatever the classifier does.

    python -m cmb.stage2.classifier --candidates ergebnisse/stage2/a03-aniso-e60-gd
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Sequence

import numpy as np

ROOT = "/home/uchralt/data/work/valdo-t2s"
CROP = (16, 32, 32)                      # (z, y, x) voxels fed to the CNN; stored patches are (20, 40, 40) or larger
JITTER = (2, 4, 4)                       # training: random offset of the crop around the patch centre, +- voxels
COARSE_CROP = (14, 28, 28)               # two-scale CNN: crop of the HALVED patch = 28 mm cube at 1 x 1 x 2 mm (needs --half 32 32 16)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# --------------------------------------------------------------------------- scoring
def lesion_scores(rows: Sequence[Dict], accepted: np.ndarray, cases: Dict[str, Dict]) -> Dict[str, float]:
    """Pooled lesion-level scores with the semantics of ``metric.hits``."""
    tp = fp = 0
    for case in cases:
        touched = set()
        for row, keep in zip(rows, accepted):
            if row["case"] != case or not keep:
                continue
            if row["touched"]:
                touched.update(row["touched"])
            else:
                fp += 1
        tp += len(touched)
    fn = sum(c["n_reference"] for c in cases.values()) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    return dict(f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0, precision=precision,
                sensitivity=sensitivity, fp_per_case=fp / len(cases), tp=tp, fp=fp, fn=fn)


def best_threshold(rows, scores, cases) -> float:
    return float(max(THRESHOLDS, key=lambda t: lesion_scores(rows, scores >= t, cases)["f1"]))


# --------------------------------------------------------------------------- hand-made features
FEATURE_NAMES = ["p_max", "p_mean_core", "log_voxels", "frst_core", "contrast", "extent_in_plane_mm",
                 "extent_through_plane_mm", "elongation", "fill_ratio", "outside_brain_fraction",
                 # only present if the candidates were built with --tta; otherwise 0 (see features_of)
                 "tta_probability", "tta_delta"]


def features_of(patch: np.ndarray, row: Dict, threshold: float) -> List[float]:
    """``patch`` is (3, z, y, x) = (image [inverted: microbleeds bright], radial symmetry, probability)."""
    from scipy import ndimage
    image, frst, prob = (patch[i].astype(np.float32) for i in range(3))
    cz, cy, cx = (s // 2 for s in image.shape)
    core = (slice(cz - 1, cz + 2), slice(cy - 2, cy + 3), slice(cx - 2, cx + 3))
    labels, _ = ndimage.label(prob > threshold, structure=np.ones((3, 3, 3), bool))
    own = labels == labels[cz, cy, cx] if labels[cz, cy, cx] else prob > threshold      # the candidate itself
    box = ndimage.find_objects(own.astype(np.uint8))[0] if own.any() else core
    dz, dy, dx = ((s.stop - s.start) for s in box)
    extent_in_plane = max(dy, dx) * 0.5                                                  # mm
    extent_through = dz * 1.0
    ring = ndimage.binary_dilation(own, iterations=6) & ~ndimage.binary_dilation(own, iterations=2) & (image > 0)
    contrast = float(image[own].mean() - image[ring].mean()) if own.any() and ring.any() else 0.0
    return [row["probability"], float(prob[core].mean()), float(np.log(row["voxels"])), float(frst[core].max()),
            contrast, extent_in_plane, extent_through, max(extent_in_plane, extent_through) / max(min(extent_in_plane, extent_through), 0.5),
            row["voxels"] / max(dz * dy * dx, 1), float((image == 0).mean()),
            # Stabilitaet unter Spiegelung: echte Blutungen aendern sich kaum, Fehlalarme stark (Faktor 7; AUC 0.711,
            # gemessen 2026-09-23). Fehlt die TTA-Karte, stehen hier Nullen -- dann traegt das Merkmal nichts bei,
            # statt den Aufruf scheitern zu lassen.
            float(row.get("tta_probability", 0.0)), float(row.get("tta_delta", 0.0))]


def fit_predict_features(kind: str) -> Callable:
    def run(train_x, train_y, test_x, seed=0):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if kind == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
        elif kind == "boosting":                     # literature scan 2026-09-20: boosted trees often beat forests on tabular features
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                   class_weight="balanced", random_state=seed)
        else:
            model = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=4)
        return model.fit(train_x, train_y).predict_proba(test_x)[:, 1]
    return run


# --------------------------------------------------------------------------- small 3D CNN
def fit_predict_cnn(train_x, train_y, test_x, seed=0, epochs=40, soft=None, alpha=0.5, crop=None):
    """``soft`` turns this into the STUDENT of a teacher-student pair (MicrobleedNet trains its discriminator by
    distillation; we never tested it -- user question 2026-09-22).

    Our adaptation, and why it is shaped this way: a teacher that sees more than the student would be cheating, and a
    teacher trained on the same candidates it then teaches would hand over memorised labels. What we do have is an
    honest source of soft targets that costs nothing extra -- the nested cross-validation of ``cmb.stage2.pooled``
    already trains an inner model per training fold and predicts the held-out training candidates with it. Those are
    out-of-fold teacher probabilities for exactly the candidates the student trains on. ``soft`` takes them and the
    student learns from a blended target ``(1 - alpha) * hard + alpha * soft``.

    What that is supposed to buy: with 106 positive VALDO candidates the hard 0/1 target forces the network to
    memorise borderline cases (a small dark dot that may or may not be a mimic) as certainties. A soft target says
    "the teacher was 0.4 sure here" and lets the student keep that doubt instead of fitting noise. ``alpha`` = 0
    reproduces the plain model exactly, 1 would drop the labels entirely; 0.5 is the untuned default.
    """
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CROP_ = tuple(crop) if crop is not None else CROP          # arm 5 step 3 (2026-09-29): e.g. (32, 32, 32) = 32 mm along z, 16 mm in plane
    assert all(c <= s for c, s in zip(CROP_, train_x.shape[-3:])), (CROP_, train_x.shape)

    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))

    def centre_crop(x):
        z0, y0, x0 = ((s - c) // 2 for s, c in zip(x.shape[-3:], CROP_))
        return x[..., z0:z0 + CROP_[0], y0:y0 + CROP_[1], x0:x0 + CROP_[2]]

    scores = np.zeros(len(test_x), np.float32)
    base = [(s - c) // 2 for s, c in zip(train_x.shape[-3:], CROP_)]
    test = torch.from_numpy(np.ascontiguousarray(centre_crop(test_x), dtype=np.float32)).to(device)      # crop first: large patches stay float16
    for member in range(3):                                   # three seeds, averaged: 160 training patches are noisy
        torch.manual_seed(seed * 10 + member); rng = np.random.RandomState(seed * 10 + member)
        net = nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64),
                            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1)).to(device)
        optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        positive_weight = torch.tensor([(train_y == 0).sum() / max((train_y == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        ziel = train_y.astype(np.float32)
        if soft is not None:                       # student: blend the hard label with the teacher's out-of-fold probability
            assert len(soft) == len(train_y), "soft targets must line up with the training candidates"
            ziel = (1 - alpha) * ziel + alpha * np.clip(np.asarray(soft, np.float32), 0.0, 1.0)
        for _ in range(epochs):
            net.train()
            order = rng.permutation(len(train_x))
            for start in range(0, len(order), 16):
                idx = order[start:start + 16]
                batch = []
                for i in idx:                                  # augmentation: random offset (+-2 / +-4 voxels) and flips
                    # around the patch centre; for the 20 x 40 x 40 patches these are the indices 0..4 / 0..8 / 0..8 as before
                    z0, y0, x0 = (b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, JITTER))
                    p = train_x[i][:, z0:z0 + CROP_[0], y0:y0 + CROP_[1], x0:x0 + CROP_[2]]
                    for axis in (1, 2, 3):
                        if rng.rand() < 0.5:
                            p = np.flip(p, axis)
                    batch.append(np.ascontiguousarray(p, dtype=np.float32))
                x = torch.from_numpy(np.stack(batch)).to(device)
                y = torch.from_numpy(ziel[idx]).to(device)
                optimiser.zero_grad(set_to_none=True)
                loss_fn(net(x).squeeze(1), y).backward()
                optimiser.step()
            schedule.step()
        net.eval()
        with torch.no_grad():
            scores += torch.sigmoid(net(test).squeeze(1)).float().cpu().numpy() / 3
    return scores


# --------------------------------------------------------------------------- self-supervised pre-training, then classify
def fit_predict_cnn_pretrained(train_x, train_y, test_x, seed=0, epochs=40, pretrain_epochs=60, mask_boxes=3):
    """The same small CNN, but its encoder is first trained WITHOUT labels to fill in masked-out regions of the patch,
    and only then fine-tuned to classify (user request 2026-09-22; the "pre-trained encoder" of the literature scan,
    shortlist #5, realised on our own data instead of downloaded weights).

    Why in-painting and not something else: the second stage is starved of labels (106 positive VALDO candidates) but
    not of images -- every cohort contributes candidate patches, and the pre-training needs none of their labels. The
    pretext task is chosen to teach the one thing that separates our two classes: a vessel CONTINUES beyond the
    lesion, a microbleed stops. To fill a masked cuboid the encoder has to learn how local structure continues, which
    is exactly that. (Intensity-style pretext tasks were deliberately not added: the robustness probe of 2026-09-22
    showed the network is already invariant to gamma, contrast and noise, so there is nothing there to learn.)

    Leak-free by construction: the pre-training sees only ``train_x``, which the caller has already restricted to the
    training folds. Nothing from the test fold enters, with or without labels.

    Cost and limitation: one pre-training per call (~1 min on an A5000), then the three members are fine-tuned from
    the SAME pre-trained encoder and differ only in head initialisation and data order -- less diverse than three
    independent members, which is a price this variant pays and which the comparison has to show is worth it.
    """
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))

    def encoder():                    # identical to the plain model, so the comparison isolates the pre-training
        return nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64))

    def decoder():                    # only used while pre-training, then thrown away
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False), block(64, 32),
                             nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False), block(32, 16),
                             nn.Conv3d(16, 3, 1))

    def centre_crop(x):
        z0, y0, x0 = ((s - c) // 2 for s, c in zip(x.shape[-3:], CROP))
        return x[..., z0:z0 + CROP[0], y0:y0 + CROP[1], x0:x0 + CROP[2]]

    def maskiere(batch: np.ndarray, rng) -> np.ndarray:
        """Zero out ``mask_boxes`` random cuboids per patch (a quarter to a half of each axis). The target stays the original."""
        out = batch.copy()
        for i in range(len(out)):
            for _ in range(mask_boxes):
                groesse = [rng.randint(max(s // 4, 1), max(s // 2, 2)) for s in CROP]
                start = [rng.randint(0, s - g + 1) for s, g in zip(CROP, groesse)]
                out[i, :, start[0]:start[0] + groesse[0], start[1]:start[1] + groesse[1], start[2]:start[2] + groesse[2]] = 0.0
        return out

    base = [(s - c) // 2 for s, c in zip(train_x.shape[-3:], CROP)]
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    enc, dec = encoder().to(device), decoder().to(device)
    optimiser = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(pretrain_epochs, 1))
    verlust = nn.MSELoss()
    for _ in range(pretrain_epochs):                                    # ---- 1) ohne Etiketten: Luecken fuellen
        enc.train(); dec.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), 16):
            idx = order[start:start + 16]
            ziel = []
            for i in idx:                                               # gleicher Versatz und gleiche Spiegelungen wie beim Klassifizieren
                shift = [b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, JITTER)]
                p = train_x[i][:, shift[0]:shift[0] + CROP[0], shift[1]:shift[1] + CROP[1], shift[2]:shift[2] + CROP[2]]
                for axis in (1, 2, 3):
                    if rng.rand() < 0.5:
                        p = np.flip(p, axis)
                ziel.append(np.ascontiguousarray(p, dtype=np.float32))
            ziel = np.stack(ziel)
            x = torch.from_numpy(maskiere(ziel, rng)).to(device)
            optimiser.zero_grad(set_to_none=True)
            verlust(dec(enc(x)), torch.from_numpy(ziel).to(device)).backward()
            optimiser.step()
        schedule.step()
    vortrainiert = {k: v.detach().clone() for k, v in enc.state_dict().items()}

    scores = np.zeros(len(test_x), np.float32)                          # ---- 2) mit Etiketten: fein abstimmen
    test = torch.from_numpy(np.ascontiguousarray(centre_crop(test_x), dtype=np.float32)).to(device)
    for member in range(3):
        torch.manual_seed(seed * 10 + member); rng = np.random.RandomState(seed * 10 + member)
        kern = encoder().to(device); kern.load_state_dict(vortrainiert)
        net = nn.Sequential(kern, nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1)).to(device)
        optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        positive_weight = torch.tensor([(train_y == 0).sum() / max((train_y == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        for _ in range(epochs):
            net.train()
            order = rng.permutation(len(train_x))
            for start in range(0, len(order), 16):
                idx = order[start:start + 16]
                batch = []
                for i in idx:
                    shift = [b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, JITTER)]
                    p = train_x[i][:, shift[0]:shift[0] + CROP[0], shift[1]:shift[1] + CROP[1], shift[2]:shift[2] + CROP[2]]
                    for axis in (1, 2, 3):
                        if rng.rand() < 0.5:
                            p = np.flip(p, axis)
                    batch.append(np.ascontiguousarray(p, dtype=np.float32))
                x = torch.from_numpy(np.stack(batch)).to(device)
                y = torch.from_numpy(train_y[idx].astype(np.float32)).to(device)
                optimiser.zero_grad(set_to_none=True)
                loss_fn(net(x).squeeze(1), y).backward()
                optimiser.step()
            schedule.step()
        net.eval()
        with torch.no_grad():
            for start in range(0, len(test_x), 128):
                teil = slice(start, start + 128)
                scores[teil] += torch.sigmoid(net(test[teil]).squeeze(1)).float().cpu().numpy() / 3
    return scores


# --------------------------------------------------------------------------- two-scale 3D CNN (fine 16 mm cube + coarse 28 mm cube)
def fit_predict_cnn_two_scale(train_x, train_y, test_x, seed=0, epochs=40):
    """The plain CNN above twice: one encoder on the central 16 mm cube at full resolution (0.5 x 0.5 x 1 mm), one on a
    28 mm cube at half resolution (1 x 1 x 2 mm); the two 64-feature vectors are concatenated before the linear layer.

    Why: the mimics are mostly vessels, and a vessel gives itself away by continuing beyond the lesion -- through the
    slices or within the plane. On cohorts with 4-5 mm slices the 16 mm cube holds only 3-4 acquired slices.
    Needs patches stored with ``cmb.stage2.candidates --half 32 32 16``. Everything else (loss, optimiser, epochs, three
    members, flips, jitter) is as in ``fit_predict_cnn`` so that the two are comparable.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert all(s >= 2 * (c + 2 * j) for s, c, j in zip(train_x.shape[-3:], COARSE_CROP, (1, 2, 2))), \
        f"patches {train_x.shape[-3:]} are too small for the coarse branch: build the candidates with --half 32 32 16"

    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))

    def encoder():
        return nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64), nn.AdaptiveAvgPool3d(1), nn.Flatten())

    class TwoScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.fine, self.coarse = encoder(), encoder()
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

        def forward(self, fine, coarse):
            return self.head(torch.cat([self.fine(fine), self.coarse(coarse)], 1))

    def halved(patches: np.ndarray) -> np.ndarray:
        """(n, 3, z, y, x) -> the same at half resolution, float32 (average pooling; done once, in chunks)."""
        out = [F.avg_pool3d(torch.from_numpy(np.ascontiguousarray(patches[i:i + 256], dtype=np.float32)), 2).numpy()
               for i in range(0, len(patches), 256)]
        return np.concatenate(out) if out else np.zeros((0, 3) + tuple(s // 2 for s in patches.shape[-3:]), np.float32)

    def window(array: np.ndarray, size, shift=(0, 0, 0)) -> np.ndarray:
        start = [(s - c) // 2 + d for s, c, d in zip(array.shape[-3:], size, shift)]
        return array[..., start[0]:start[0] + size[0], start[1]:start[1] + size[1], start[2]:start[2] + size[2]]

    train_coarse, test_coarse = halved(train_x), halved(test_x)
    test_fine = torch.from_numpy(np.ascontiguousarray(window(test_x, CROP), dtype=np.float32))
    test_coarse = torch.from_numpy(np.ascontiguousarray(window(test_coarse, COARSE_CROP)))
    scores = np.zeros(len(test_x), np.float32)
    for member in range(3):
        torch.manual_seed(seed * 10 + member); rng = np.random.RandomState(seed * 10 + member)
        net = TwoScale().to(device)
        optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        positive_weight = torch.tensor([(train_y == 0).sum() / max((train_y == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        for _ in range(epochs):
            net.train()
            order = rng.permutation(len(train_x))
            for start in range(0, len(order), 16):
                idx = order[start:start + 16]
                fine, coarse = [], []
                for i in idx:                                  # one random offset for both scales (coarse: half of it), same flips
                    shift = [int(rng.randint(-j, j + 1)) for j in JITTER]
                    a = window(train_x[i], CROP, shift)
                    b = window(train_coarse[i], COARSE_CROP, [d // 2 for d in shift])
                    for axis in (1, 2, 3):
                        if rng.rand() < 0.5:
                            a, b = np.flip(a, axis), np.flip(b, axis)
                    fine.append(np.ascontiguousarray(a, dtype=np.float32)); coarse.append(np.ascontiguousarray(b, dtype=np.float32))
                y = torch.from_numpy(train_y[idx].astype(np.float32)).to(device)
                optimiser.zero_grad(set_to_none=True)
                loss_fn(net(torch.from_numpy(np.stack(fine)).to(device), torch.from_numpy(np.stack(coarse)).to(device)).squeeze(1), y).backward()
                optimiser.step()
            schedule.step()
        net.eval()
        with torch.no_grad():
            for start in range(0, len(test_x), 128):
                part = slice(start, start + 128)
                scores[part] += torch.sigmoid(net(test_fine[part].to(device), test_coarse[part].to(device)).squeeze(1)).float().cpu().numpy() / 3
    return scores


# --------------------------------------------------------------------------- nested cross-validation
def nested_cv(name: str, fit_predict: Callable, inputs: np.ndarray, rows: List[Dict], cases: Dict[str, Dict]) -> Dict:
    labels = np.array([1 if r["touched"] else 0 for r in rows])
    folds = np.array([r["fold"] for r in rows])
    accepted = np.zeros(len(rows), bool)
    chosen = []
    for k in range(5):
        train = np.nonzero(folds != k)[0]; test = np.nonzero(folds == k)[0]
        # inner loop: out-of-fold scores within the training folds -> acceptance threshold
        inner_scores = np.zeros(len(train), np.float32)
        for j in sorted(set(folds[train])):
            fit = train[folds[train] != j]; held = folds[train] == j
            inner_scores[held] = fit_predict(inputs[fit], labels[fit], inputs[train[held]], seed=k * 10 + j)
        train_cases = {c: v for c, v in cases.items() if v["fold"] != k}
        threshold = best_threshold([rows[i] for i in train], inner_scores, train_cases)
        chosen.append(threshold)
        accepted[test] = fit_predict(inputs[train], labels[train], inputs[test], seed=k) >= threshold
    result = lesion_scores(rows, accepted, cases)
    per_fold = [round(lesion_scores([r for r in rows if r["fold"] == k], accepted[folds == k],
                                    {c: v for c, v in cases.items() if v["fold"] == k})["f1"], 3) for k in range(5)]
    print(f"{name:26s} F1 {result['f1']:.3f}  precision {result['precision']:.2f}  sensitivity {result['sensitivity']:.2f}  "
          f"FP/case {result['fp_per_case']:.2f}  per fold {per_fold}  thresholds {chosen}", flush=True)
    return dict(result, per_fold=per_fold, thresholds=chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--models", nargs="*", default=["probability", "logistic", "forest", "cnn"])
    args = parser.parse_args()
    folder = args.candidates if os.path.isabs(args.candidates) else f"{ROOT}/{args.candidates}"
    info = json.load(open(f"{folder}/candidates.json"))
    rows, cases = info["candidates"], info["cases"]
    patches = np.load(f"{folder}/patches.npy")
    print(f"{len(rows)} candidates, {sum(bool(r['touched']) for r in rows)} on a microbleed; "
          f"{sum(c['n_reference'] for c in cases.values())} reference lesions in {len(cases)} cases")

    everything = lesion_scores(rows, np.ones(len(rows), bool), cases)
    print(f"{'accept every candidate':26s} F1 {everything['f1']:.3f}  precision {everything['precision']:.2f}  "
          f"sensitivity {everything['sensitivity']:.2f}  FP/case {everything['fp_per_case']:.2f}")
    report = {"accept_all": everything}
    feats = np.array([features_of(patches[i], rows[i], info["threshold"]) for i in range(len(rows))], np.float32)
    runners = {
        # baseline: no second stage, only a (nested) threshold on the stage-1 probability
        "probability": (lambda tx, ty, sx, seed=0: sx[:, 0], feats),
        "logistic": (fit_predict_features("logistic"), feats),
        "forest": (fit_predict_features("forest"), feats),
        "boosting": (fit_predict_features("boosting"), feats),
        "cnn": (fit_predict_cnn, patches),
    }
    for name in args.models:
        fit_predict, inputs = runners[name]
        report[name] = nested_cv({"probability": "stage-1 probability only"}.get(name, "stage 2: " + name), fit_predict, inputs, rows, cases)
    if "forest" in args.models:                      # which features carry the decision? (all data, for interpretation only)
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=600, min_samples_leaf=3, class_weight="balanced", random_state=0)
        forest.fit(feats, [1 if r["touched"] else 0 for r in rows])
        ranking = sorted(zip(FEATURE_NAMES, forest.feature_importances_), key=lambda t: -t[1])
        report["feature_importance"] = {n: round(float(v), 3) for n, v in ranking}
        print("feature importance:", ", ".join(f"{n} {v:.2f}" for n, v in ranking))
    json.dump(report, open(f"{folder}/stage2_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- pre-train on one cohort, fine-tune on another
def fit_predict_cnn_transfer(source_x, source_y, finetune_x, finetune_y, test_x, seed=0, epochs=40, finetune_epochs=20,
                             finetune_lr=3e-4, cache=None):
    """The plain CNN (same layers, crop, jitter, flips and three averaged members as ``fit_predict_cnn``), but trained in
    two steps (user idea 2026-09-28, ARBEIT.md 5ae arm 4): PRE-TRAINED on the candidates of a source cohort (VALDO, public),
    then FINE-TUNED on the candidates of the target cohort with a smaller learning rate. With an empty fine-tuning set the
    pre-trained network is used as is (zero-shot transfer of the classifier). Why: the pooled classifier needs the private
    candidates at training time and cannot be published; this recipe can -- U-Net and classifier trained on VALDO,
    a clinic re-trains its own U-Net and only tunes the classifier on a few local candidates.
    ``cache`` (dict) keeps the pre-trained weights per seed: the source set does not depend on the target fold, so
    pre-training once per seed is exact, not an approximation."""
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))

    def new_net():
        return nn.Sequential(block(3, 16), nn.MaxPool3d(2), block(16, 32), nn.MaxPool3d(2), block(32, 64),
                             nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1)).to(device)

    def centre_crop(x):
        z0, y0, x0 = ((s - c) // 2 for s, c in zip(x.shape[-3:], CROP))
        return x[..., z0:z0 + CROP[0], y0:y0 + CROP[1], x0:x0 + CROP[2]]

    def train(net, xs, ys, member_seed, n_epochs, lr):
        rng = np.random.RandomState(member_seed)
        base = [(s - c) // 2 for s, c in zip(xs.shape[-3:], CROP)]
        optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
        positive_weight = torch.tensor([(ys == 0).sum() / max((ys == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        ziel = ys.astype(np.float32)
        for _ in range(n_epochs):
            net.train()
            order = rng.permutation(len(xs))
            for start in range(0, len(order), 16):
                idx = order[start:start + 16]
                batch = []
                for i in idx:
                    z0, y0, x0 = (b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, JITTER))
                    p = xs[i][:, z0:z0 + CROP[0], y0:y0 + CROP[1], x0:x0 + CROP[2]]
                    for axis in (1, 2, 3):
                        if rng.rand() < 0.5:
                            p = np.flip(p, axis)
                    batch.append(np.ascontiguousarray(p, dtype=np.float32))
                x = torch.from_numpy(np.stack(batch)).to(device)
                y = torch.from_numpy(ziel[idx]).to(device)
                optimiser.zero_grad(set_to_none=True)
                loss_fn(net(x).squeeze(1), y).backward()
                optimiser.step()
            schedule.step()
        return net

    scores = np.zeros(len(test_x), np.float32)
    test = torch.from_numpy(np.ascontiguousarray(centre_crop(test_x), dtype=np.float32)).to(device)
    for member in range(3):
        member_seed = seed * 10 + member
        key = ("pre", member_seed)
        if cache is not None and key in cache:
            net = new_net(); net.load_state_dict(cache[key])
        else:
            torch.manual_seed(member_seed)
            net = train(new_net(), source_x, source_y, member_seed, epochs, 1e-3)
            if cache is not None:
                cache[key] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if len(finetune_x):
            torch.manual_seed(member_seed + 7)
            net = train(net, finetune_x, finetune_y, member_seed + 7, finetune_epochs, finetune_lr)
        net.eval()
        with torch.no_grad():
            scores += torch.sigmoid(net(test).squeeze(1)).float().cpu().numpy() / 3
    return scores
