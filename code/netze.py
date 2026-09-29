#!/usr/bin/env python3
"""Turnier 18.09.2026 (Claude, Nutzerauftrag): zehn moeglichst verschiedene U-Net-Architekturen
fuer CMB auf VALDO T2*, alle mit derselben Schnittstelle

    Eingang (B, nch, D, H, W) in (z, y, x)  ->  Logits (B, 1, D, H, W)

damit code/turnier.py sie in den unveraenderten cv5-Ablauf (Falten, Stichprobe, innere Wahl von
Epoche/Schwelle, Klumpen-Waechter, metric.hits) einsetzen kann. Netz 10 (zweistufig) ist kein
Einzelnetz und steht in code/stufe2.py; es setzt auf die Karten des besten Netzes von hier auf.

Warum je Netz, in einem Satz (ausfuehrlich: protokoll.md, Eintrag "Turnier"):
  a01-attunet   Messlatte: das bisherige 3D-Attention-U-Net (= cv5 r3, InstanceNorm).
  a02-nnunet    schlichtes U-Net im nnU-Net-Stil (LeakyReLU, InstanceNorm, Deep Supervision):
                bringen die Gates ueberhaupt etwas?
  a03-aniso     erste Stufe faltet/poolt nur in der Schicht: Gitter C ist 0.5x0.5x1.0 mm, und
                Kohorte 1 hat nativ 4-mm-Schichten -- durch die Schicht steckt wenig echte Information.
  a04-2p5d      2D-U-Net, 5 Nachbarschichten als Kanaele: viel mehr Trainingsbeispiele je Fall.
  a05-resse     Residual-Bloecke + Squeeze-Excitation: tiefer, stabile Gradienten, Kanalgewichtung
                (T2* gegen FRST).
  a06-unetpp    U-Net++ (dichte, verschachtelte Skips): Objekte von wenigen Voxeln verschwinden
                beim Herunterskalieren; die dichten Skips halten die feine Aufloesung.
  a07-flach     nur 2 Poolingstufen, dilatierte Faltungen in der Mitte: eine CMB ist 2-10 mm,
                wenige Parameter fuer 57 Faelle.
  a08-swin      Swin-UNETR (Transformer-Hybrid), wenn vorhanden mit vortrainiertem Encoder:
                globaler Kontext (Lage, Symmetrie von Verkalkungen). Der andersartigste Kandidat.
  a09-segres    SegResNet: grosser Encoder, kleiner Decoder, GroupNorm -- fuer wenig Daten gebaut.

Normierung: ueberall InstanceNorm (Befund 11.09.: BatchNorm bei Stapel 4 mit 60 % Laesions-
Patches passt nicht zu den Inferenzkacheln), ausser a09 (GroupNorm gehoert zur Architektur).

Selbsttest (CPU, jede Architektur einmal vor und zurueck, ungerade Patchgroesse):
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
    """Polstert (D,H,W) auf Vielfache von `vielfaches` (mindestens `mindest`) und schneidet die
    Ausgabe zurueck. `waehle` holt aus Listen-/Tupelausgaben (U-Net++) den Tensor."""
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
    """(Conv -> InstanceNorm -> LeakyReLU) * 2 mit waehlbarem Kern/Dilatation."""
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
    """Stufe 0 arbeitet nur in der Schicht (Kern 1x3x3, Pooling 1x2x2). Danach ist das Gitter
    1.0 mm isotrop und die Stufen sind gewoehnlich. z wird also einmal weniger halbiert."""
    def __init__(self, nch, basis=32, gates=False):
        super().__init__()
        b = basis
        # gates=True (a11, 19.09.): dieselben Attention-Gates wie a01 auf allen vier Skips --
        # beantwortet, ob Attention AUF dem Turniersieger etwas bringt.
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
    """Jede z-Schicht wird mit ihren 2*r Nachbarn als Kanaele von einem 2D-U-Net segmentiert.
    (B, C, D, H, W) -> (B*D, C*(2r+1), H, W) -> (B, 1, D, H, W). Rand in z: Spiegelung der
    Randschicht (replicate), nicht Null -- Null saehe aus wie ein Signalabfall."""
    def __init__(self, nch, r=2, basis=32):
        super().__init__()
        self.r = r
        self.netz = UNet2D(nch * (2 * r + 1), basis=basis, tiefe=4)

    def forward(self, x):
        B, C, D, H, W = x.shape
        r = self.r
        xp = F.pad(x, (0, 0, 0, 0, r, r), mode="replicate")               # z polstern
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
    """Encoder mit je zwei ResSE-Bloecken (tiefer als das Doppel-Conv-U-Net), Decoder mit einem."""
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
    """Zwei Poolingstufen (kleinstes Gitter 2x2x4 mm). Die Mitte erweitert das Sichtfeld mit
    Dilatation 1/2/4 statt mit weiterem Herunterskalieren -- die Aufloesung bleibt erhalten."""
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
    """DynUNet = die nnU-Net-Architektur in MONAI. Im Training liefert es (B, Koepfe, 1, D, H, W)
    (Deep Supervision, alle Koepfe schon auf volle Groesse interpoliert); turnier.py gewichtet
    die Koepfe 1, 1/2, 1/4. In eval() kommt nur der Hauptkopf."""
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
    """Swin-UNETR, feature_size 48 (so gross sind die vortrainierten Gewichte). Der vortrainierte
    Encoder (selbstueberwacht auf 5050 CT, 1 Kanal) wird geladen, wenn die Datei da ist; die
    Eingangsfaltung wird dann auf nch Kanaele verteilt (Gewicht / nch je Kanal). Ob geladen
    wurde, steht in n.vortrainiert und landet in meta.json -- kein stilles Entweder-Oder."""
    from monai.networks.nets import SwinUNETR
    n = SwinUNETR(in_channels=nch, out_channels=1, feature_size=48, use_checkpoint=True, spatial_dims=3)
    geladen = 0
    roh = {}
    if os.path.exists(VORTRAINIERT_SWIN):
        try:
            roh = torch.load(VORTRAINIERT_SWIN, map_location="cpu", weights_only=False)
            roh = roh.get("state_dict", roh)
        except Exception as e:                              # halb geladene Datei (instabile Leitung)
            print(f"Swin: vortrainierte Gewichte unlesbar ({type(e).__name__}) -- Start von null", flush=True)
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


# ---------------------------------------------------------------- Register

# ---------------------------------------------------------------- Turnier 2 (29.09.2026): Breite, rekurrentes 2D-Netz, MedNeXt, nnU-Net mit Residual-Encoder
class RekurrentBlock2D(nn.Module):
    """R2U-Net-Baustein (Alom et al. 2018): Residual + rekurrente Faltung (t Wiederholungen mit geteilten Gewichten)."""
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
    """a04 mit rekurrent-residualem 2D-U-Net (R2U-Net) statt des schlichten 2D-U-Nets."""
    def __init__(self, nch, r=2, basis=32):
        super().__init__(nch, r=r, basis=basis)
        self.netz = R2UNet2D(nch * (2 * r + 1), basis=basis, tiefe=4)


def baue_mednext(nch):
    """MedNeXt-S (Roy et al. 2023) aus MONAI 1.5: ConvNeXt-Bloecke im U-Net, Kern 3, ohne Deep Supervision."""
    from monai.networks.nets import MedNeXt
    n = MedNeXt(spatial_dims=3, init_filters=32, in_channels=nch, out_channels=1, encoder_expansion_ratio=2, decoder_expansion_ratio=2,
                bottleneck_expansion_ratio=2, kernel_size=3, deep_supervision=False, use_residual_connection=True, blocks_down=(2, 2, 2, 2),
                blocks_bottleneck=2, blocks_up=(2, 2, 2, 2), norm_type="group", global_resp_norm=False)
    return Polster(n, (16, 16, 16))


def baue_nnunet_resenc(nch):
    """Wie a02-nnunet (DynUNet), aber mit Residual-Bloecken im Encoder (nnU-Net ResEnc-Voreinstellung)."""
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
    "a11-anisoatt": lambda nch: Polster(AnisoUNet3D(nch, gates=True), (8, 16, 16)),   # Sieger + Attention-Gates
    # Turnier 2 (29.09.2026)
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
    x = torch.randn(1, 2, 38, 62, 94)                       # die echte (ungerade) Patchgroesse
    for name in namen:
        t0 = time.time()
        m = NETZE[name](2).train()
        y = m(x)
        if y.dim() == 6:                                    # Deep Supervision
            assert y.shape[2:] == (1, 38, 62, 94), (name, y.shape)
            y.mean().backward()
            y = m.eval()(x)
        else:
            y.mean().backward()
        assert y.shape == (1, 1, 38, 62, 94), (name, y.shape)
        ohne_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        print(f"{name:12s} {parameterzahl(m) / 1e6:6.2f} Mio Parameter, Ausgabe {tuple(y.shape)}, "
              f"ohne Gradient {len(ohne_grad)}, vortrainiert {getattr(m, 'vortrainiert', '-')}, "
              f"{time.time() - t0:.1f}s", flush=True)
