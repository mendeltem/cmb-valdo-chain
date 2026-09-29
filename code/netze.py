#!/usr/bin/env python3
"""Tournament 18 Sep 2026 (Claude, user request): ten as-different-as-possible U-Net architectures
for CMB on VALDO T2*, all with the same interface

    Input (B, nch, D, H, W) in (z, y, x)  ->  logits (B, 1, D, H, W)

so that code/turnier.py can plug them into the unchanged cv5 flow (folds, sampling, inner choice of
epoch/threshold, blob guard, metric.hits). Net 10 (two-stage) is not a
single network and lives in code/stufe2.py; it builds on the maps of the best network from here.

Why each network, in one sentence (in detail: protokoll.md, entry "Turnier"):
  a01-attunet   benchmark: the previous 3D attention U-Net (= cv5 r3, InstanceNorm).
  a02-nnunet    plain U-Net in nnU-Net style (LeakyReLU, InstanceNorm, deep supervision):
                do the gates bring anything at all?
  a03-aniso     first stage convolves/pools only within the slice: Grid C is 0.5x0.5x1.0 mm, and
                cohort 1 natively has 4 mm slices -- little real information lies through the slice.
  a04-2p5d      2D U-Net, 5 neighbouring slices as channels: far more training examples per case.
  a05-resse     residual blocks + squeeze-excitation: deeper, stable gradients, channel weighting
                (T2* against FRST).
  a06-unetpp    U-Net++ (dense, nested skips): objects of only a few voxels vanish
                when downscaling; the dense skips preserve the fine resolution.
  a07-flach     only 2 pooling stages, dilated convolutions in the middle: a CMB is 2-10 mm,
                few parameters for 57 cases.
  a08-swin      Swin-UNETR (transformer hybrid), with a pretrained encoder if available:
                global context (position, symmetry of calcifications). The most dissimilar candidate.
  a09-segres    SegResNet: large encoder, small decoder, GroupNorm -- built for little data.

Normalisation: InstanceNorm throughout (finding 11 Sep: BatchNorm at batch size 4 with 60% lesion
patches doesn't fit the inference tiles), except a09 (GroupNorm belongs to the architecture).

Self-test (CPU, every architecture once forward and backward, odd patch size):
    python netze.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_unet import AttentionUNet3D, AttentionGate3D

VORTRAINIERT_SWIN = "/home/uchralt/data/work/valdo-t2s/dev/vortrainiert/model_swinvit.pt"


# ---------------------------------------------------------------- Helfer
class Polster(nn.Module):
    """Pads (D,H,W) to multiples of `vielfaches` (at least `mindest`) and crops the
    output back. `waehle` picks the tensor out of list/tuple outputs (U-Net++)."""
    def __init__(self, netz, vielfaches, mindest=(0, 0, 0), waehle=None):
        super().__init__()
        self.netz, self.v, self.m, self.waehle = netz, tuple(vielfaches), tuple(mindest), waehle

    def forward(self, x):
        g = x.shape[-3:]
        ziel = [max(-(-g[i] // self.v[i]) * self.v[i], self.m[i]) for i in range(3)]
        pd = [ziel[i] - g[i] for i in range(3)]
        if any(pd):
            x = F.pad(x, (0, pd[2], 0, pd[1], 0, pd[0]))
        y = self.netz(x)
        if self.waehle is not None:
            y = self.waehle(y)
        return y[..., :g[0], :g[1], :g[2]]


def inorm(c):
    return nn.InstanceNorm3d(c, affine=True)


class Doppel(nn.Module):
    """(Conv -> InstanceNorm -> LeakyReLU) * 2 with selectable kernel/dilation."""
    def __init__(self, ein, aus, kern=(3, 3, 3), dil=1):
        super().__init__()
        pad = tuple(dil * (k // 2) for k in kern)
        self.net = nn.Sequential(
            nn.Conv3d(ein, aus, kern, padding=pad, dilation=dil, bias=False), inorm(aus), nn.LeakyReLU(0.01, True),
            nn.Conv3d(aus, aus, kern, padding=pad, dilation=dil, bias=False), inorm(aus), nn.LeakyReLU(0.01, True))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- a03: anisotrop
class AnisoUNet3D(nn.Module):
    """Stage 0 operates only within the slice (kernel 1x3x3, pooling 1x2x2). After that the grid
    is 1.0 mm isotropic and the stages are ordinary. z is therefore halved one time fewer."""
    def __init__(self, nch, basis=32, gates=False):
        super().__init__()
        b = basis
        # gates=True (a11, 19 Sep): the same attention gates as a01 on all four skips --
        # answers whether attention ON TOP OF the tournament winner brings anything.
        self.ag = nn.ModuleList([AttentionGate3D(c, c, norm="instance") for c in (8 * b, 4 * b, 2 * b, b)]) if gates else None
        self.e0 = Doppel(nch, b, kern=(1, 3, 3))
        self.e1, self.e2, self.e3 = Doppel(b, 2 * b), Doppel(2 * b, 4 * b), Doppel(4 * b, 8 * b)
        self.mitte = Doppel(8 * b, 16 * b)
        self.p0, self.p = nn.MaxPool3d((1, 2, 2)), nn.MaxPool3d(2)
        self.u3 = nn.ConvTranspose3d(16 * b, 8 * b, 2, stride=2)
        self.u2 = nn.ConvTranspose3d(8 * b, 4 * b, 2, stride=2)
        self.u1 = nn.ConvTranspose3d(4 * b, 2 * b, 2, stride=2)
        self.u0 = nn.ConvTranspose3d(2 * b, b, (1, 2, 2), stride=(1, 2, 2))
        self.d3, self.d2, self.d1 = Doppel(16 * b, 8 * b), Doppel(8 * b, 4 * b), Doppel(4 * b, 2 * b)
        self.d0 = Doppel(2 * b, b, kern=(1, 3, 3))
        self.aus = nn.Conv3d(b, 1, 1)

    def forward(self, x):
        s0 = self.e0(x); s1 = self.e1(self.p0(s0)); s2 = self.e2(self.p(s1)); s3 = self.e3(self.p(s2))
        h = self.mitte(self.p(s3))
        for i, (u, d, s) in enumerate(((self.u3, self.d3, s3), (self.u2, self.d2, s2),
                                       (self.u1, self.d1, s1), (self.u0, self.d0, s0))):
            g = u(h)
            if self.ag is not None:
                s = self.ag[i](g, s)
            h = d(torch.cat([g, s], 1))
        return self.aus(h)


# ---------------------------------------------------------------- a04: 2.5D
class UNet2D(nn.Module):
    def __init__(self, ein, basis=32, tiefe=4):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.InstanceNorm2d(b, affine=True),
                                 nn.LeakyReLU(0.01, True),
                                 nn.Conv2d(b, b, 3, padding=1, bias=False), nn.InstanceNorm2d(b, affine=True),
                                 nn.LeakyReLU(0.01, True))
        ch = [basis * 2 ** i for i in range(tiefe + 1)]
        self.e = nn.ModuleList([blk(ein, ch[0])] + [blk(ch[i], ch[i + 1]) for i in range(tiefe)])
        self.u = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(tiefe))])
        self.d = nn.ModuleList([blk(2 * ch[i], ch[i]) for i in reversed(range(tiefe))])
        self.aus = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.e):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for i, (u, d) in enumerate(zip(self.u, self.d)):
            x = d(torch.cat([u(x), skips[-2 - i]], 1))
        return self.aus(x)


class ZweiEinhalbD(nn.Module):
    """Each z-slice is segmented with its 2*r neighbours as channels by a 2D U-Net.
    (B, C, D, H, W) -> (B*D, C*(2r+1), H, W) -> (B, 1, D, H, W). Edge in z: mirroring of the
    edge slice (replicate), not zero -- zero would look like a signal drop-off."""
    def __init__(self, nch, r=2, basis=32):
        super().__init__()
        self.r = r
        self.netz = UNet2D(nch * (2 * r + 1), basis=basis, tiefe=4)

    def forward(self, x):
        B, C, D, H, W = x.shape
        r = self.r
        xp = F.pad(x, (0, 0, 0, 0, r, r), mode="replicate")               # pad z
        st = xp.unfold(2, 2 * r + 1, 1)                                   # (B, C, D, H, W, 2r+1)
        st = st.permute(0, 2, 1, 5, 3, 4).reshape(B * D, C * (2 * r + 1), H, W)
        pd = ((16 - H % 16) % 16, (16 - W % 16) % 16)
        if any(pd):
            st = F.pad(st, (0, pd[1], 0, pd[0]))
        y = self.netz(st)[..., :H, :W]                                    # (B*D, 1, H, W)
        return y.reshape(B, D, 1, H, W).permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------- a05: Residual + SE
class ResSE(nn.Module):
    def __init__(self, ein, aus, r=8):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv3d(ein, aus, 3, padding=1, bias=False), inorm(aus), nn.LeakyReLU(0.01, True))
        self.c2 = nn.Sequential(nn.Conv3d(aus, aus, 3, padding=1, bias=False), inorm(aus))
        self.se = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(aus, max(aus // r, 4), 1), nn.ReLU(True),
                                nn.Conv3d(max(aus // r, 4), aus, 1), nn.Sigmoid())
        self.kurz = nn.Identity() if ein == aus else nn.Sequential(nn.Conv3d(ein, aus, 1, bias=False), inorm(aus))

    def forward(self, x):
        h = self.c2(self.c1(x))
        return F.leaky_relu(h * self.se(h) + self.kurz(x), 0.01)


class ResSEUNet3D(nn.Module):
    """Encoder with two ResSE blocks each (deeper than the double-conv U-Net), decoder with one."""
    def __init__(self, nch, basis=32, tiefe=4):
        super().__init__()
        ch = [basis * 2 ** i for i in range(tiefe + 1)]
        self.e = nn.ModuleList([nn.Sequential(ResSE(nch, ch[0]), ResSE(ch[0], ch[0]))] +
                               [nn.Sequential(ResSE(ch[i], ch[i + 1]), ResSE(ch[i + 1], ch[i + 1]))
                                for i in range(tiefe)])
        self.u = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(tiefe))])
        self.d = nn.ModuleList([ResSE(2 * ch[i], ch[i]) for i in reversed(range(tiefe))])
        self.aus = nn.Conv3d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.e):
            x = e(x if i == 0 else F.max_pool3d(x, 2))
            skips.append(x)
        for i, (u, d) in enumerate(zip(self.u, self.d)):
            x = d(torch.cat([u(x), skips[-2 - i]], 1))
        return self.aus(x)


# ---------------------------------------------------------------- a07: flach + dilatiert
class FlachUNet3D(nn.Module):
    """Two pooling stages (smallest grid 2x2x4 mm). The middle expands the field of view with
    dilation 1/2/4 instead of further downscaling -- the resolution is preserved."""
    def __init__(self, nch, basis=32):
        super().__init__()
        b = basis
        self.e0, self.e1 = Doppel(nch, b), Doppel(b, 2 * b)
        self.m1, self.m2, self.m4 = Doppel(2 * b, 4 * b, dil=1), Doppel(4 * b, 4 * b, dil=2), Doppel(4 * b, 4 * b, dil=4)
        self.misch = nn.Sequential(nn.Conv3d(12 * b, 4 * b, 1, bias=False), inorm(4 * b), nn.LeakyReLU(0.01, True))
        self.u1 = nn.ConvTranspose3d(4 * b, 2 * b, 2, stride=2)
        self.u0 = nn.ConvTranspose3d(2 * b, b, 2, stride=2)
        self.d1, self.d0 = Doppel(4 * b, 2 * b), Doppel(2 * b, b)
        self.aus = nn.Conv3d(b, 1, 1)

    def forward(self, x):
        s0 = self.e0(x); s1 = self.e1(F.max_pool3d(s0, 2))
        a = self.m1(F.max_pool3d(s1, 2)); b_ = self.m2(a); c = self.m4(b_)
        h = self.misch(torch.cat([a, b_, c], 1))
        h = self.d1(torch.cat([self.u1(h), s1], 1))
        h = self.d0(torch.cat([self.u0(h), s0], 1))
        return self.aus(h)


# ---------------------------------------------------------------- MONAI-Netze
def baue_nnunet(nch):
    """DynUNet = the nnU-Net architecture in MONAI. In training it returns (B, heads, 1, D, H, W)
    (deep supervision, all heads already interpolated to full size); turnier.py weights
    the heads 1, 1/2, 1/4. In eval() only the main head comes out."""
    from monai.networks.nets import DynUNet
    n = DynUNet(spatial_dims=3, in_channels=nch, out_channels=1,
                kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2], upsample_kernel_size=[2, 2, 2, 2],
                filters=[32, 64, 128, 256, 320], norm_name=("instance", {"affine": True}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                deep_supervision=True, deep_supr_num=2, res_block=False)
    return Polster(n, (16, 16, 16))


def baue_unetpp(nch):
    from monai.networks.nets import BasicUNetPlusPlus
    n = BasicUNetPlusPlus(spatial_dims=3, in_channels=nch, out_channels=1,
                          features=(32, 32, 64, 128, 256, 32), deep_supervision=False,
                          norm=("instance", {"affine": True}), act=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
    return Polster(n, (16, 16, 16), waehle=lambda y: y[0] if isinstance(y, (list, tuple)) else y)


def baue_swin(nch):
    """Swin-UNETR, feature_size 48 (that's how big the pretrained weights are). The pretrained
    encoder (self-supervised on 5050 CT, 1 channel) is loaded if the file is present; the
    input convolution is then spread across nch channels (weight / nch per channel). Whether it was
    loaded is recorded in n.vortrainiert and lands in meta.json -- no silent either/or."""
    from monai.networks.nets import SwinUNETR
    n = SwinUNETR(in_channels=nch, out_channels=1, feature_size=48, use_checkpoint=True, spatial_dims=3)
    geladen = 0
    roh = {}
    if os.path.exists(VORTRAINIERT_SWIN):
        try:
            roh = torch.load(VORTRAINIERT_SWIN, map_location="cpu", weights_only=False)
            roh = roh.get("state_dict", roh)
        except Exception as e:                              # half-loaded file (unstable connection)
            print(f"Swin: pretrained weights unreadable ({type(e).__name__}) -- starting from scratch", flush=True)
            roh = {}
    if roh:
        eigen = n.swinViT.state_dict()
        for k, v in roh.items():
            k2 = k.replace("module.", "").replace("swin_vit.", "").replace("fc", "linear")
            if k2 in eigen:
                if eigen[k2].shape == v.shape:
                    eigen[k2] = v; geladen += 1
                elif k2.endswith("patch_embed.proj.weight") and v.shape[1] == 1:
                    eigen[k2] = v.repeat(1, nch, 1, 1, 1) / nch; geladen += 1
        n.swinViT.load_state_dict(eigen)
    w = Polster(n, (32, 32, 32), mindest=(64, 64, 64))
    w.vortrainiert = geladen
    return w


def baue_segres(nch):
    from monai.networks.nets import SegResNet
    n = SegResNet(spatial_dims=3, init_filters=32, in_channels=nch, out_channels=1,
                  blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.1)
    return Polster(n, (8, 8, 8))


# ---------------------------------------------------------------- Registry

# ---------------------------------------------------------------- Tournament 2 (29 Sep 2026): wide, recurrent 2D net, MedNeXt, nnU-Net with residual encoder
class RekurrentBlock2D(nn.Module):
    """R2U-Net building block (Alom et al. 2018): residual + recurrent convolution (t repetitions with shared weights)."""
    def __init__(self, ein, aus, t=2):
        super().__init__()
        self.t = t
        self.ein = nn.Conv2d(ein, aus, 1, bias=False)
        self.f1 = nn.Sequential(nn.Conv2d(aus, aus, 3, padding=1, bias=False), nn.InstanceNorm2d(aus, affine=True), nn.LeakyReLU(0.01, True))
        self.f2 = nn.Sequential(nn.Conv2d(aus, aus, 3, padding=1, bias=False), nn.InstanceNorm2d(aus, affine=True), nn.LeakyReLU(0.01, True))

    def forward(self, x):
        x = self.ein(x)
        h = self.f1(x)
        for _ in range(self.t - 1):
            h = self.f1(x + h)
        g = self.f2(h)
        for _ in range(self.t - 1):
            g = self.f2(h + g)
        return x + g


class R2UNet2D(nn.Module):
    def __init__(self, ein, basis=32, tiefe=4, t=2):
        super().__init__()
        ch = [basis * 2 ** i for i in range(tiefe + 1)]
        self.e = nn.ModuleList([RekurrentBlock2D(ein, ch[0], t)] + [RekurrentBlock2D(ch[i], ch[i + 1], t) for i in range(tiefe)])
        self.u = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(tiefe))])
        self.d = nn.ModuleList([RekurrentBlock2D(2 * ch[i], ch[i], t) for i in reversed(range(tiefe))])
        self.aus = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.e):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for i, (u, d) in enumerate(zip(self.u, self.d)):
            x = d(torch.cat([u(x), skips[-2 - i]], 1))
        return self.aus(x)


class ZweiEinhalbDR2(ZweiEinhalbD):
    """a04 with a recurrent-residual 2D U-Net (R2U-Net) instead of the plain 2D U-Net."""
    def __init__(self, nch, r=2, basis=32):
        super().__init__(nch, r=r, basis=basis)
        self.netz = R2UNet2D(nch * (2 * r + 1), basis=basis, tiefe=4)


def baue_mednext(nch):
    """MedNeXt-S (Roy et al. 2023) from MONAI 1.5: ConvNeXt blocks in the U-Net, kernel 3, without deep supervision."""
    from monai.networks.nets import MedNeXt
    n = MedNeXt(spatial_dims=3, init_filters=32, in_channels=nch, out_channels=1, encoder_expansion_ratio=2, decoder_expansion_ratio=2,
                bottleneck_expansion_ratio=2, kernel_size=3, deep_supervision=False, use_residual_connection=True, blocks_down=(2, 2, 2, 2),
                blocks_bottleneck=2, blocks_up=(2, 2, 2, 2), norm_type="group", global_resp_norm=False)
    return Polster(n, (16, 16, 16))


def baue_nnunet_resenc(nch):
    """Like a02-nnunet (DynUNet), but with residual blocks in the encoder (nnU-Net ResEnc preset)."""
    from monai.networks.nets import DynUNet
    n = DynUNet(spatial_dims=3, in_channels=nch, out_channels=1,
                kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2], upsample_kernel_size=[2, 2, 2, 2],
                filters=[32, 64, 128, 256, 320], norm_name=("instance", {"affine": True}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}), deep_supervision=True, deep_supr_num=2, res_block=True)
    return Polster(n, (16, 16, 16))


NETZE = {
    "a01-attunet": lambda nch: AttentionUNet3D(in_channels=nch, basis=32, tiefe=4, gates=True, norm="instance"),
    "a02-nnunet":  baue_nnunet,
    "a03-aniso":   lambda nch: Polster(AnisoUNet3D(nch), (8, 16, 16)),
    "a04-2p5d":    lambda nch: ZweiEinhalbD(nch, r=2),
    "a05-resse":   lambda nch: Polster(ResSEUNet3D(nch), (16, 16, 16)),
    "a06-unetpp":  baue_unetpp,
    "a07-flach":   lambda nch: Polster(FlachUNet3D(nch), (4, 4, 4)),
    "a08-swin":    baue_swin,
    "a09-segres":  baue_segres,
    "a11-anisoatt": lambda nch: Polster(AnisoUNet3D(nch, gates=True), (8, 16, 16)),   # winner + attention gates
    # Tournament 2 (29 Sep 2026)
    "a03-aniso-b16": lambda nch: Polster(AnisoUNet3D(nch, basis=16), (8, 16, 16)),
    "a03-aniso-b48": lambda nch: Polster(AnisoUNet3D(nch, basis=48), (8, 16, 16)),
    "a13-r2unet2p5d": lambda nch: ZweiEinhalbDR2(nch, r=2),
    "a14-mednext":  baue_mednext,
    "a15-nnunet-resenc": baue_nnunet_resenc,
}


def parameterzahl(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import sys, time
    torch.manual_seed(0)
    namen = sys.argv[1:] or list(NETZE)
    x = torch.randn(1, 2, 38, 62, 94)                       # the real (odd) patch size
    for name in namen:
        t0 = time.time()
        m = NETZE[name](2).train()
        y = m(x)
        if y.dim() == 6:                                    # deep supervision
            assert y.shape[2:] == (1, 38, 62, 94), (name, y.shape)
            y.mean().backward()
            y = m.eval()(x)
        else:
            y.mean().backward()
        assert y.shape == (1, 1, 38, 62, 94), (name, y.shape)
        ohne_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        print(f"{name:12s} {parameterzahl(m) / 1e6:6.2f} M parameters, output {tuple(y.shape)}, "
              f"without gradient {len(ohne_grad)}, pretrained {getattr(m, 'vortrainiert', '-')}, "
              f"{time.time() - t0:.1f}s", flush=True)
