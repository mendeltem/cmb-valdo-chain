"""TrackSet-CMB: a second-stage classifier assembled from switchable modules (2026-10-25).

User request 2026-10-25: "completely new mixed models, U-Net plus 3D classifier, that nobody had before" -- after a
literature audit (wiki cmb_detection/search-space-expansion-and-novel-hybrids-2026-10). Why stage 2: with a perfect
classifier the chain would reach F1 ~0.85 on VALDO instead of ~0.65, and on FIXED candidates the seed spread of stage 2 is
about ten times smaller than that of a U-Net training (VALDO sd 0.009), so one arm costs ~1 GPU hour and small effects
become visible. Every module is a switch, so each can be measured alone against the same training loop without it:

  d4      in-plane D4 group convolutions instead of plain 3D convolutions for the local crop (exactly invariant to
          rotations by 90 degrees and mirror images in the slice plane). Why: 3D G-CNNs reach the FROC of a CNN trained on
          ten times more data in nodule FP reduction (Winkels 2018, arXiv 1804.04656); we have ~100 positive VALDO
          candidates. Full 3D rotation groups are wrong here: slices are 0.8-5 mm thick, the grid is 1 x 0.5 x 0.5 mm.
  track   learned slice track: from the candidate centre, follow the bright (inverted: dark on T2*) structure slice by slice
          up and down, take a small 2D crop centred on the tracked point in every slice, and let a small transformer read
          the sequence (crop, in-plane drift, contrast). Question it answers: does the structure continue as a vessel or end
          as a blob? A learned form of the hand rule 'run length along z' (Arm 5, 2026-09-29: AUC 0.72 on one external SWI
          set, neutral elsewhere). No such model was found for CMB or other small lesions (prior-art check 2026-10-25).
  ctx     soft context: tissue class (5 one-hot), stage-1 probability and log size as inputs, instead of the hard CSF rule
          that deletes real deep CMB on SWI.
  set     one attention layer over all candidates of the SAME scan, with a learned distance penalty, so candidates on one
          vessel or in one heavy-burden scan can inform each other (port of NoduleSAT, arXiv 2004.05640; not found for CMB).
  atlas   ctx plus three venous-atlas values at the candidate (cmb.stage2.atlas_features, Arena 2 round 3 R5); needs candidate
          folders built with those fields.
  proto   prototype margin loss on the candidate embedding (CMB prototype vs mimic prototype), after the Hard Sample
          Prototype Learning of Kim 2025 (Med Image Anal, DOI 10.1016/j.media.2024.103415). Every negative candidate is a
          mined stage-1 false positive, i.e. exactly the 'hard sample' of that paper.

Controls for d4 (literature check 2026-10-25, scratchpad F_equivariant: the G-CNN gain shrinks with data, and augmentation
plus test-time averaging can match it, Gerken 2022): every arm trains with random D4 + z-flip augmentation; 'tta8' averages
the prediction over the 8 in-plane views at test time; 'wide' widens the plain CNN to 40-80-160 channels (about the
compute of the D4 encoder). The d4 arm counts only if it beats tset-tta8 and tset-wide-tta8.
The plain arm (no module) is the small 3D CNN of cmb.stage2.classifier.fit_predict_cnn rebuilt inside this training loop,
so every module is compared against a control that differs ONLY by that module.

Inputs come as a Bundle (patches + side features + case index + centre in mm + optional tracks); cmb.stage2.pooled builds
it. Models are listed in MODELS; pooled exposes them as --models tset-plain, tset-d4, ...
"""
from __future__ import annotations

import functools
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

VOXEL_MM_XYZ = np.array([0.5, 0.5, 1.0])      # centre_voxel of cmb.stage2.candidates is (x, y, z) on the model grid
INPLANE_MM = 0.5                               # patches are (channel, z, y, x) on the same grid: z 1 mm, y/x 0.5 mm
CROP = (16, 32, 32)                            # local crop (z, y, x), as cmb.stage2.classifier.CROP
JITTER = (2, 4, 4)
TRACK_HALF = 9                                 # slices followed up and down (9 mm each on the 1 mm grid)
TRACK_CROP = 15                                # odd: the centre pixel stays the centre under rotations and flips
TRACK_RADIUS = 3                               # search radius per slice in voxels (1.5 mm in-plane per 1 mm in z: ~56 degrees)
SIDE_FEATURES = 9                              # tissue one-hot (5), stage-1 probability, log voxels, vote count, brain-edge distance
ATLAS_FEATURES = 3                             # appended after SIDE_FEATURES; used only by module 'atlas'
EDGE_CAP_MM = 10.0


# --------------------------------------------------------------------------- inputs
class Bundle:
    """Per-candidate inputs that can be sliced like a numpy array (cmb.stage2.pooled.nested does inputs[index])."""

    def __init__(self, patches, side, case_index, centre_mm, tracks: Optional[Dict[str, np.ndarray]] = None):
        self.patches = patches
        self.side = np.asarray(side, np.float32)
        self.case_index = np.asarray(case_index, np.int64)
        self.centre_mm = np.asarray(centre_mm, np.float32)
        self.tracks = tracks

    def __len__(self) -> int:
        return len(self.case_index)

    def __getitem__(self, index) -> "Bundle":
        index = np.asarray(index)
        if index.dtype == bool:
            index = np.nonzero(index)[0]
        tracks = None if self.tracks is None else {k: v[index] for k, v in self.tracks.items()}
        return Bundle(self.patches[index], self.side[index], self.case_index[index], self.centre_mm[index], tracks)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self),) + tuple(self.patches.shape[1:])


def brain_edge_mm(patches: np.ndarray, cap_mm: float = EDGE_CAP_MM) -> np.ndarray:
    """Distance (mm) from the patch centre to the nearest voxel outside the brain (image channel 0 == 0; the grids are masked
    and cmb.stage2.candidates pads outside the volume with 0), capped at cap_mm. Why (Arena 2 round 3, R9): candidates within
    ~5 mm of the brain edge are mostly sulcal veins, but a hard 5 mm rule costs 7-8 points of true-positive rate -- so the
    distance enters stage 2 as a learned feature."""
    from scipy import ndimage
    out = np.full(len(patches), cap_mm, np.float32)
    for i in range(len(patches)):
        brain = np.asarray(patches[i][0], np.float32) > 0
        if not brain.all():
            c = tuple(s // 2 for s in brain.shape)
            out[i] = min(float(ndimage.distance_transform_edt(brain, sampling=(1.0, INPLANE_MM, INPLANE_MM))[c]), cap_mm)
    return out


def side_from_rows(rows: Sequence[Dict], patches: np.ndarray, tissue_class: Dict[int, int]) -> np.ndarray:
    """(n, SIDE_FEATURES + ATLAS_FEATURES) float32:
       tissue class one-hot (5, from the SynthSeg majority label), stage-1 probability, log size scaled to about unit range,
       vote count = (n_runs - 1) / 4 clipped to [0, 1] (cmb.stage2.merge_candidates: how many stage-1 runs found the object; 0 for
       a single run -- the LUNA16 lesson, Arena 2 round 3 R8: keep the vote as an explicit feature), brain-edge distance / 10 mm,
       then three atlas values (cmb.stage2.atlas_features; zeros when absent): vein probability at the centre, its maximum within
       3 mm, distance to the nearest vein-probability >= 0.5 voxel / 20 mm."""
    n = len(rows)
    side = np.zeros((n, SIDE_FEATURES + ATLAS_FEATURES), np.float32)
    for i, r in enumerate(rows):
        k = tissue_class.get(r.get("tissue", -1))
        if k is not None:
            side[i, k] = 1.0
        side[i, 5] = r["probability"]
        side[i, 6] = (np.log(r["voxels"]) - 3.0) / 2.0
        side[i, 7] = min(max(r.get("n_runs", 1) - 1, 0), 4) / 4.0
        side[i, 9] = r.get("atlas_vein_p", 0.0)
        side[i, 10] = r.get("atlas_vein_max3mm", 0.0)
        side[i, 11] = min(r.get("atlas_vein_dist_mm", 20.0), 20.0) / 20.0
    side[:, 8] = brain_edge_mm(patches) / EDGE_CAP_MM
    return side


# --------------------------------------------------------------------------- slice tracking (numpy, CPU)
def track_candidates(patches: np.ndarray, half: int = TRACK_HALF, size: int = TRACK_CROP, radius: int = TRACK_RADIUS,
                     sigma: float = 1.0) -> Dict[str, np.ndarray]:
    """Follow the bright structure (image channel 0 is inverted: microbleeds and vessels bright) through the slices.

    -> crops (n, 2*half+1, 3, size, size) float16: all three channels, centred on the tracked point of every slice;
       feats (n, 2*half+1, 5) float32: drift dy, dx in mm from the centre track point, contrast (tracked value minus the
              median of a ring of 4-7 voxels on the smoothed slice), valid (1 inside the patch), relative z in units of 10 mm.
    Slices outside the stored patch stay zero with valid = 0. The search moves at most `radius` voxels per slice, so a
    vessel oblique up to ~56 degrees stays centred; a blob that ends leaves the track wandering on noise with low contrast.
    """
    from scipy import ndimage
    n, _, Z, Y, X = patches.shape
    L = 2 * half + 1
    crops = np.zeros((n, L, 3, size, size), np.float16)
    feats = np.zeros((n, L, 5), np.float32)
    h = size // 2
    yy, xx = np.mgrid[-7:8, -7:8]
    ring = (np.hypot(yy, xx) >= 4) & (np.hypot(yy, xx) <= 7)
    feats[:, :, 4] = (np.arange(L) - half) / 10.0
    for i in range(n):
        vol = np.asarray(patches[i], np.float32)
        smooth = ndimage.gaussian_filter(vol[0], (0, sigma, sigma))
        zc, yc, xc = Z // 2, Y // 2, X // 2

        def best(z, y, x):
            y0, y1 = max(y - radius, 0), min(y + radius + 1, Y)
            x0, x1 = max(x - radius, 0), min(x + radius + 1, X)
            window = smooth[z, y0:y1, x0:x1]
            j = int(np.argmax(window))
            return y0 + j // window.shape[1], x0 + j % window.shape[1]

        start = best(zc, yc, xc)
        for direction in (0, 1, -1):
            y, x = start
            steps = [0] if direction == 0 else range(1, half + 1)
            for step in steps:
                z = zc + direction * step
                slot = half + direction * step
                if not 0 <= z < Z:
                    break
                if step:
                    y, x = best(z, y, x)
                for c in range(3):
                    crops[i, slot, c] = _crop2d(vol[c, z], y, x, h)
                r = _crop2d(smooth[z], y, x, 7)[ring]
                feats[i, slot, :4] = ((y - start[0]) * INPLANE_MM, (x - start[1]) * INPLANE_MM,
                                      float(smooth[z, y, x] - np.median(r)), 1.0)
    return dict(crops=crops, feats=feats)


def _crop2d(image: np.ndarray, y: int, x: int, h: int) -> np.ndarray:
    out = np.zeros((2 * h + 1, 2 * h + 1), np.float32)
    y0, y1, x0, x1 = y - h, y + h + 1, x - h, x + h + 1
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, image.shape[0]), max(x0, 0), min(x1, image.shape[1])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


# --------------------------------------------------------------------------- neighbours (CPU, for the feature models)
NEIGHBOUR_NAMES = ["log_candidates_in_scan", "neighbours_10mm", "max_p_neighbours_10mm", "collinear_between_15mm",
                   "mean_dist_3nn_mm"]


def neighbour_features(case_index: np.ndarray, centre_mm: np.ndarray, p_max: np.ndarray) -> np.ndarray:
    """(n, 5) features from the OTHER candidates of the same scan -- a cheap proxy for the set module.

    collinear_between_15mm: max over pairs of neighbours within 15 mm of -cos(angle) between the two vectors from the
    candidate to them; 1 = the candidate sits on a straight line between two others (beads on a vessel), 0 = no pair.
    """
    out = np.zeros((len(case_index), 5), np.float32)
    for case in np.unique(case_index):
        idx = np.nonzero(case_index == case)[0]
        pts = centre_mm[idx]
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        for a, i in enumerate(idx):
            near = np.nonzero(d[a] <= 10.0)[0]
            close = np.nonzero(d[a] <= 15.0)[0]
            collinear = 0.0
            if len(close) >= 2:
                v = pts[close] - pts[a]
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                cos = v @ v.T
                np.fill_diagonal(cos, 1.0)
                collinear = float(max(0.0, -cos.min()))
            finite = np.sort(d[a][np.isfinite(d[a])])[:3]
            out[i] = (np.log1p(len(idx) - 1), len(near), float(p_max[idx[near]].max()) if len(near) else 0.0, collinear,
                      float(min(finite.mean(), 50.0)) if len(finite) else 50.0)
    return out


# --------------------------------------------------------------------------- D4 (in-plane rotations by 90 degrees and mirrors)
def d4_apply(t, g: int):
    """Element g = 0..7 of D4 on the last two axes: mirror x first (g >= 4), then rotate by (g % 4) * 90 degrees."""
    import torch
    if g >= 4:
        t = torch.flip(t, (-1,))
    return torch.rot90(t, g % 4, (-2, -1))


@functools.lru_cache(maxsize=1)
def d4_tables() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    """(product table T[a][b] with d4_apply(d4_apply(x, b), a) == d4_apply(x, T[a][b]), inverse table), found numerically
    -- so the group convolution cannot get the convention of the composition wrong."""
    import torch
    x = torch.arange(25.0).reshape(5, 5)
    ops = [d4_apply(x, g) for g in range(8)]
    table = tuple(tuple(next(c for c in range(8) if torch.equal(d4_apply(d4_apply(x, b), a), ops[c])) for b in range(8))
                  for a in range(8))
    inverse = tuple(next(b for b in range(8) if table[a][b] == 0) for a in range(8))
    return table, inverse


def d4_offset_matrix(g: int) -> np.ndarray:
    """2 x 2 integer matrix M with (dy, dx)_new = M @ (dy, dx) for a point relative to the centre of an odd-sized image."""
    import torch
    cols = []
    for dy, dx in ((1, 0), (0, 1)):
        img = torch.zeros(5, 5)
        img[2 + dy, 2 + dx] = 1
        y, x = (int(v) for v in torch.nonzero(d4_apply(img, g))[0])
        cols.append((y - 2, x - 2))
    return np.array(cols, np.float32).T


def _modules():
    """Torch modules live in a function so that importing this file needs no torch (CPU tools import trackset)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class D4Conv3d(nn.Module):
        """Group convolution over D4 acting in the (y, x) plane; z is untouched. lift=True: plain input (B, C, Z, Y, X) ->
        (B, F * 8, Z, Y, X); otherwise (B, C * 8, ...) -> (B, F * 8, ...). Channel index = field * 8 + group element."""

        def __init__(self, c_in: int, c_out: int, kernel=3, lift: bool = False):
            super().__init__()
            self.lift, self.c_in, self.c_out = lift, c_in, c_out
            shape = (c_out, c_in, 3, kernel, kernel) if lift else (c_out, c_in, 8, 3, kernel, kernel)
            self.weight = nn.Parameter(torch.empty(shape))
            nn.init.kaiming_uniform_(self.weight.view(c_out, -1, 3, kernel, kernel), a=0.01)
            table, inverse = d4_tables()
            # group conv: W_g[:, :, h] = op_g(W[:, :, g^-1 h])
            self.register_buffer("perm", torch.tensor([[table[inverse[g]][h] for h in range(8)] for g in range(8)]), persistent=False)

        def full_weight(self):
            if self.lift:
                ws = [d4_apply(self.weight, g) for g in range(8)]                          # each (F, C, 3, k, k)
                w = torch.stack(ws, 1)                                                     # (F, 8, C, 3, k, k)
                return w.reshape(self.c_out * 8, self.c_in, *self.weight.shape[-3:])
            ws = [d4_apply(self.weight[:, :, self.perm[g]], g) for g in range(8)]          # each (F, C, 8, 3, k, k)
            w = torch.stack(ws, 1)                                                         # (F, 8, C, 8, 3, k, k)
            return w.reshape(self.c_out * 8, self.c_in * 8, *self.weight.shape[-3:])

        def forward(self, x):
            k = self.weight.shape[-1]
            return F.conv3d(x, self.full_weight(), padding=(1, k // 2, k // 2))

    class FieldNorm(nn.Module):
        """InstanceNorm per channel without affine, then one scale and shift per FIELD (shared by its 8 group channels):
        instance statistics move with the channel permutation, so the layer stays equivariant."""

        def __init__(self, fields: int):
            super().__init__()
            self.norm = nn.InstanceNorm3d(fields * 8, affine=False)
            self.scale = nn.Parameter(torch.ones(fields)); self.shift = nn.Parameter(torch.zeros(fields))
            self.fields = fields

        def forward(self, x):
            b, _, *s = x.shape
            x = self.norm(x).view(b, self.fields, 8, *s)
            return (x * self.scale.view(1, -1, 1, 1, 1, 1) + self.shift.view(1, -1, 1, 1, 1, 1)).view(b, self.fields * 8, *s)

    def d4_block(c_in, c_out, lift=False):
        return nn.Sequential(D4Conv3d(c_in, c_out, lift=lift), FieldNorm(c_out), nn.LeakyReLU(0.01, True),
                             D4Conv3d(c_out, c_out), FieldNorm(c_out), nn.LeakyReLU(0.01, True))

    class D4Encoder(nn.Module):
        """Fields 8-16-24 (64-128-192 group channels, ~0.3 M parameters, the plain CNN has 0.22 M). Max pooling 2 on even
        sizes keeps exact equivariance; group mean + global mean make the 24-d output invariant."""
        out_dim = 24

        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(d4_block(3, 8, lift=True), nn.MaxPool3d(2), d4_block(8, 16), nn.MaxPool3d(2), d4_block(16, 24))

        def forward(self, x):
            h = self.body(x)
            b = h.shape[0]
            return h.view(b, 24, 8, -1).mean((2, 3))

    def plain_block(c_in, c_out):
        return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True),
                             nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), nn.InstanceNorm3d(c_out, affine=True), nn.LeakyReLU(0.01, True))

    class PlainEncoder(nn.Module):
        """The network of cmb.stage2.classifier.fit_predict_cnn without its last linear layer (widths 16-32-64);
        wide=True: 40-80-160, about the compute of the D4 encoder (control arm)."""

        def __init__(self, wide: bool = False, c_in: int = 3):
            super().__init__()
            a, b, c = (40, 80, 160) if wide else (16, 32, 64)
            self.out_dim = c
            self.body = nn.Sequential(plain_block(c_in, a), nn.MaxPool3d(2), plain_block(a, b), nn.MaxPool3d(2), plain_block(b, c),
                                      nn.AdaptiveAvgPool3d(1), nn.Flatten())

        def forward(self, x):
            return self.body(x)

    class TrackEncoder(nn.Module):
        """Shared 2D CNN per tracked slice crop, then a 2-layer transformer over the slices; mean over valid slices."""
        out_dim = 48

        def __init__(self):
            super().__init__()
            def b2(ci, co):
                return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.01, True))
            self.cnn = nn.Sequential(b2(3, 16), b2(16, 16), nn.MaxPool2d(2), b2(16, 32), b2(32, 32), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.token = nn.Sequential(nn.Linear(32 + 5, 48), nn.LayerNorm(48))
            layer = nn.TransformerEncoderLayer(48, 4, 96, dropout=0.2, batch_first=True)
            self.seq = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)   # padded slices: no nested fast path

        def forward(self, crops, feats):
            b, L = crops.shape[:2]
            e = self.cnn(crops.reshape(b * L, *crops.shape[2:])).view(b, L, -1)
            t = self.token(torch.cat([e, feats], -1))
            valid = feats[..., 3] > 0.5
            h = self.seq(t, src_key_padding_mask=~valid)
            h = torch.nan_to_num(h) * valid[..., None]
            return h.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)

    class TrackSetNet(nn.Module):
        def __init__(self, modules: Sequence[str], c_in: int = 3):
            super().__init__()
            self.use = set(modules)
            self.local = D4Encoder() if "d4" in self.use else PlainEncoder(wide="wide" in self.use, c_in=c_in)
            width = self.local.out_dim
            if "track" in self.use:
                self.track = TrackEncoder(); width += TrackEncoder.out_dim
            if "ctx" in self.use or "atlas" in self.use:
                self.n_side = SIDE_FEATURES + (ATLAS_FEATURES if "atlas" in self.use else 0)
                self.ctx = nn.Sequential(nn.Linear(self.n_side, 16), nn.LeakyReLU(0.01, True)); width += 16
            self.fuse = nn.Sequential(nn.Dropout(0.3), nn.Linear(width, 64), nn.LeakyReLU(0.01, True))
            if "set" in self.use:
                self.attn = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
                self.alpha = nn.Parameter(torch.tensor(0.0))                     # distance penalty, softplus >= 0
                self.set_norm = nn.LayerNorm(64)
            self.head = nn.Linear(64, 1)
            self.prototypes = nn.Parameter(torch.randn(2, 64) * 0.1)            # 0 = mimic, 1 = microbleed

        def forward(self, crop, side, case, centre, track_crops=None, track_feats=None):
            parts = [self.local(crop)]
            if "track" in self.use:
                parts.append(self.track(track_crops, track_feats))
            if "ctx" in self.use or "atlas" in self.use:
                parts.append(self.ctx(side[:, :self.n_side]))
            z = self.fuse(torch.cat(parts, 1))
            if "set" in self.use:
                same = case[:, None] == case[None, :]
                dist = torch.cdist(centre, centre) / 10.0
                bias = -F.softplus(self.alpha) * dist
                bias = bias.masked_fill(~same, float("-inf"))
                a, _ = self.attn(z[None], z[None], z[None], attn_mask=bias)
                z = self.set_norm(z + a[0])
            return self.head(z).squeeze(1), z

    return dict(TrackSetNet=TrackSetNet, D4Encoder=D4Encoder, PlainEncoder=PlainEncoder, D4Conv3d=D4Conv3d, TrackEncoder=TrackEncoder)


def prototype_loss(z, y, prototypes, margin: float = 0.2):
    """Normalised margin loss: (d_own - d_other) / (d_own + d_other) + margin, clipped at 0 (Kim 2025, concentration loss)."""
    import torch
    d = torch.cdist(z, prototypes) ** 2                                    # (n, 2)
    own = d.gather(1, y.long()[:, None]).squeeze(1)
    other = d.gather(1, (1 - y.long())[:, None]).squeeze(1)
    return torch.relu((own - other) / (own + other + 1e-6) + margin).mean()


# --------------------------------------------------------------------------- training
def _batches_by_case(case_index: np.ndarray, rng: np.random.RandomState, target: int = 32, cap: int = 0):
    """Whole scans per batch (the set module needs all candidates of a scan together), about `target` candidates each.
    cap > 0 (training): a scan with more candidates than `cap` contributes a random subset of `cap` per epoch -- heavy-burden
    scans have 100+ candidates and would otherwise blow the GPU memory of the D4 encoder (~20 MB of activations each)."""
    cases = rng.permutation(np.unique(case_index))
    batch, size = [], 0
    for c in cases:
        idx = np.nonzero(case_index == c)[0]
        if cap and len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        batch.append(idx); size += len(idx)
        if size >= target:
            yield np.concatenate(batch); batch, size = [], 0
    if batch:
        yield np.concatenate(batch)


def fit_predict_trackset(train_x: Bundle, train_y: np.ndarray, test_x: Bundle, seed: int = 0, modules: Sequence[str] = (),
                         epochs: int = 40, members: int = 3, proto_weight: float = 0.5, device: Optional[str] = None,
                         channels: Sequence[int] = (0, 1, 2),
                         pretrain_x: Optional[Bundle] = None, pretrain_y: Optional[np.ndarray] = None,
                         finetune_epochs: Optional[int] = None, finetune_lr: Optional[float] = None,
                         pretrain_cache: Optional[dict] = None) -> np.ndarray:
    """Same contract as cmb.stage2.classifier.fit_predict_cnn: train on train_x, return probabilities for test_x.
    Three members with different seeds are averaged, as in the plain CNN.

    channels (wk2 mechanism arms, 2026-10-29): which PATCH channels the local encoder reads. Default (0,1,2) =
    image + FRST + probability, bit-identical to before. (1,2) = FRST + probability (C-pf), (2,) = probability only
    (C-prob). The patch channel layout is (image, frst, probability) from cmb.stage2.candidates."""
    import torch
    import torch.nn as nn
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # repeatable on the GPU too (user 2026-10-25: "everything repeatable"): deterministic cuDNN kernels, no autotuning
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net_cls = _modules()["TrackSetNet"]
    modules = tuple(modules)
    need_track = "track" in modules
    if need_track:
        assert train_x.tracks is not None and test_x.tracks is not None, "module 'track' needs Bundle.tracks (track_candidates)"
    if tuple(channels) != (0, 1, 2):
        assert "d4" not in modules and "wide" not in modules, "channel subsets are for the plain encoder only"
    assert all(c <= s for c, s in zip(CROP, train_x.patches.shape[-3:])), (CROP, train_x.patches.shape)
    base = [(s - c) // 2 for s, c in zip(train_x.patches.shape[-3:], CROP)]
    offset_mats = [torch.tensor(d4_offset_matrix(g)) for g in range(8)]

    def make_batch(bundle: Bundle, idx, rng=None, view: int = 0):
        """rng given: training batch with random offset, D4 element and z-flip per candidate; otherwise the centred crop
        seen through the fixed D4 element `view` (test-time averaging over the 8 views with module 'tta8')."""
        ch = tuple(channels)
        crops = []
        for i in idx:
            if rng is None:
                z0, y0, x0 = base
            else:
                z0, y0, x0 = (b - j + rng.randint(0, 2 * j + 1) for b, j in zip(base, JITTER))
            crops.append(np.asarray(bundle.patches[i][list(ch), z0:z0 + CROP[0], y0:y0 + CROP[1], x0:x0 + CROP[2]], np.float32))
        crop = torch.from_numpy(np.stack(crops)).to(device)
        tc = tf = None
        if need_track:
            tc = torch.from_numpy(np.asarray(bundle.tracks["crops"][idx], np.float32)).to(device)
            tf = torch.from_numpy(np.asarray(bundle.tracks["feats"][idx], np.float32)).to(device)
        if rng is not None:                                   # one D4 element and one z-flip per candidate, applied to ALL its inputs
            for j in range(len(idx)):
                g = int(rng.randint(8)); flip_z = rng.rand() < 0.5
                crop[j] = d4_apply(crop[j], g)
                if flip_z:
                    crop[j] = torch.flip(crop[j], (1,))
                if need_track:
                    tc[j] = d4_apply(tc[j], g)
                    tf[j, :, :2] = tf[j, :, :2] @ offset_mats[g].to(device).T
                    if flip_z:
                        tc[j] = torch.flip(tc[j], (0,)); tf[j] = torch.flip(tf[j], (0,)); tf[j, :, 4] = -tf[j, :, 4]
        elif view:
            crop = d4_apply(crop, view)
            if need_track:
                tc = d4_apply(tc, view)
                tf[:, :, :2] = tf[:, :, :2] @ offset_mats[view].to(device).T
        side = torch.from_numpy(bundle.side[idx]).to(device)
        case = torch.from_numpy(bundle.case_index[idx]).to(device)
        centre = torch.from_numpy(bundle.centre_mm[idx]).to(device)
        return crop, side, case, centre, tc, tf

    def run_epochs(net, xs, ys, member_seed, n_epochs, lr):
        """One training phase on bundle xs (identical to the original single-phase loop: same rng
        order, same batches, same schedule)."""
        rng = np.random.RandomState(member_seed)
        optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
        positive_weight = torch.tensor([(ys == 0).sum() / max((ys == 1).sum(), 1)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        for _ in range(n_epochs):
            net.train()
            for idx in _batches_by_case(xs.case_index, rng, cap=96):
                crop, side, case, centre, tc, tf = make_batch(xs, idx, rng)
                y = torch.from_numpy(ys[idx].astype(np.float32)).to(device)
                logit, z = net(crop, side, case, centre, tc, tf)
                loss = loss_fn(logit, y)
                if "proto" in modules:
                    loss = loss + proto_weight * prototype_loss(z, y, net.prototypes)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            schedule.step()

    scores = np.zeros(len(test_x), np.float32)
    for member in range(members):
        torch.manual_seed(seed * 10 + member); rng = np.random.RandomState(seed * 10 + member)
        net = net_cls(modules, c_in=len(tuple(channels))).to(device)
        if pretrain_x is not None:
            # wk2b P->FT: first phase on the pooled pre-training set (full epochs/lr), then the fine-tuning
            # phase on the target cohort's own set with the (smaller) fine-tuning budget. The pre-trained
            # weights are cached per (pre-set, member seed, channels): the pre-set only depends on the held-out
            # fold, so every inner/test fit of that fold reuses the exact same pre-trained net.
            ck = (len(pretrain_x.patches), int(np.asarray(pretrain_y).sum()), seed * 10 + member, tuple(channels))
            if pretrain_cache is not None and ck in pretrain_cache:
                net.load_state_dict(pretrain_cache[ck])
            else:
                run_epochs(net, pretrain_x, pretrain_y, seed * 10 + member, epochs, 1e-3)
                if pretrain_cache is not None:
                    pretrain_cache[ck] = net.state_dict()
            run_epochs(net, train_x, train_y, seed * 10 + member, finetune_epochs, finetune_lr)
        else:
            run_epochs(net, train_x, train_y, seed * 10 + member, epochs, 1e-3)
        net.eval()
        with torch.no_grad():
            views = range(8) if "tta8" in modules else [0]
            for idx in _batches_by_case(test_x.case_index, np.random.RandomState(0), target=256):
                for view in views:
                    logit, _ = net(*make_batch(test_x, idx, view=view))
                    scores[idx] += torch.sigmoid(logit).float().cpu().numpy() / (members * len(views))
    return scores


ARMS = {"tset-plain": (), "tset-tta8": ("tta8",), "tset-wide-tta8": ("wide", "tta8"), "tset-d4": ("d4",), "tset-d4proto": ("d4", "proto"), "tset-track": ("track",),
        "tset-ctx": ("ctx",), "tset-atlas": ("ctx", "atlas"), "tset-set": ("set",), "tset-proto": ("proto",),
        "tset-full": ("d4", "track", "ctx", "set", "proto")}
MODELS = {name: functools.partial(fit_predict_trackset, modules=mods) for name, mods in ARMS.items()}
