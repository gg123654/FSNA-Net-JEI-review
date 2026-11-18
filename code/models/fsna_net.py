"""FSNA-Net: Frequency-Spatial Noise-Aware Network for Low-Light Enhancement.

IGAB family: Illumination-Guided Attention Block as SRB (RetinexFormer-style
transposed IG-MSA), keep IE / FRB / NAF, no FullImageGainHead.
Params target: 1.5–6M (aim ~3–5M).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=True)


def conv1x1(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, 1, 0, bias=True)


class DWConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True)
        self.pw = conv1x1(channels, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            conv3x3(channels, channels),
            nn.GELU(),
            conv3x3(channels, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, mid),
            nn.GELU(),
            conv1x1(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class IllumGuidedBlock(nn.Module):
    """CNN illumination-guided residual (used by FRB refine / out head)."""

    def __init__(self, channels: int):
        super().__init__()
        self.v = nn.Sequential(conv3x3(channels, channels), nn.GELU(), conv3x3(channels, channels))
        self.gate = nn.Sequential(
            conv3x3(channels + 1, channels),
            nn.GELU(),
            conv1x1(channels, channels),
            nn.Sigmoid(),
        )
        self.ca = ChannelAttention(channels)
        self.proj = conv1x1(channels, channels)

    def forward(self, x: torch.Tensor, l_map: torch.Tensor) -> torch.Tensor:
        if l_map.shape[-2:] != x.shape[-2:]:
            l_map = F.interpolate(l_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        v = self.v(x)
        g = self.gate(torch.cat([x, l_map], dim=1))
        return self.proj(x + self.ca(v * g))


# ---------------------------------------------------------------------------
# IGAB (Illumination-Guided Attention Block) — transposed MSA, O(C^2)
# ---------------------------------------------------------------------------


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,W,C]
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IG_MSA(nn.Module):
    """Illumination-Guided Multi-head Self-Attention (transposed, O(C^2))."""

    def __init__(self, dim: int, dim_head: int = 32, heads: int = 1):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        inner = dim_head * heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(inner, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def _split_heads(self, t: torch.Tensor, b: int) -> torch.Tensor:
        # [B,N,H*D] -> [B,H,N,D]
        n = t.shape[1]
        t = t.view(b, n, self.num_heads, self.dim_head)
        return t.permute(0, 2, 1, 3)

    def forward(self, x_in: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
        # x_in / illu: [B,H,W,C]
        b, h, w, c = x_in.shape
        n = h * w
        x = x_in.reshape(b, n, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu = illu_fea_trans.reshape(b, n, c)
        # project illu to same inner dim as v via viewing as multi-head (must match)
        # Use linear-free broadcast: map through same layout as v (c must equal heads*dim_head
        # when dim==heads*dim_head; otherwise project with 1x1 via Linear on last dim).
        if illu.shape[-1] != q_inp.shape[-1]:
            # fallback: channel-align by linear built on the fly is avoided — require match
            raise RuntimeError(f"illu channels {illu.shape[-1]} != attn inner {q_inp.shape[-1]}")
        q = self._split_heads(q_inp, b)
        k = self._split_heads(k_inp, b)
        v = self._split_heads(v_inp, b)
        illu_h = self._split_heads(illu, b)
        v = v * illu_h
        # transposed attention: [B,H,D,N]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B,H,D,N]
        out = out.permute(0, 3, 1, 2).reshape(b, n, self.num_heads * self.dim_head)
        out_c = self.proj(out).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.view(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class IGAB(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, heads: int = 1, num_blocks: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        # x, illu_fea: [B,C,H,W]
        if illu_fea.shape[-2:] != x.shape[-2:]:
            illu_fea = F.interpolate(illu_fea, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if illu_fea.shape[1] != x.shape[1]:
            raise RuntimeError(f"illu_fea C={illu_fea.shape[1]} != x C={x.shape[1]}")
        x = x.permute(0, 2, 3, 1)
        illu = illu_fea.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class IlluminationEstimator(nn.Module):
    """Predict light-up map E≥1 and multi-channel illu features for IGAB."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            conv1x1(4, channels),
            nn.GELU(),
        )
        self.depth = nn.Conv2d(channels, channels, 5, 1, 2, groups=min(4, channels), bias=True)
        self.map_head = nn.Sequential(
            conv3x3(channels, channels),
            nn.GELU(),
            DWConv(channels),
            ResBlock(channels),
            conv3x3(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prior = x.max(dim=1, keepdim=True).values
        h = self.stem(torch.cat([x, prior], dim=1))
        illu_fea = self.depth(h)
        e = 1.0 + F.softplus(self.map_head(illu_fea))
        e = e.clamp(1.0, 20.0)
        l_hat = (1.0 / e).clamp(1e-3, 1.0)
        return e, l_hat, illu_fea


class SpatialRestorationBranch(nn.Module):
    """IGAB U-Net spatial restorer guided by illumination features."""

    def __init__(
        self,
        in_ch: int = 7,
        base: int = 32,
        level: int = 2,
        num_blocks: tuple[int, ...] = (1, 2, 2),
    ):
        super().__init__()
        assert level >= 1
        assert len(num_blocks) == level + 1
        self.level = level
        self.base = base
        self.embedding = nn.Conv2d(in_ch, base, 3, 1, 1, bias=False)
        self.illu_proj = nn.Conv2d(base, base, 1, 1, 0, bias=False)

        self.encoder_layers = nn.ModuleList()
        dim = base
        for i in range(level):
            heads = max(dim // base, 1)
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(dim=dim, dim_head=base, heads=heads, num_blocks=num_blocks[i]),
                        nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim *= 2

        heads_b = max(dim // base, 1)
        self.bottleneck = IGAB(dim=dim, dim_head=base, heads=heads_b, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList()
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim, dim // 2, 2, 2),
                        nn.Conv2d(dim, dim // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim // 2,
                            dim_head=base,
                            heads=max((dim // 2) // base, 1),
                            num_blocks=num_blocks[level - 1 - i],
                        ),
                    ]
                )
            )
            dim //= 2

        self.mapping = nn.Conv2d(base, 3, 3, 1, 1, bias=False)

    def _align(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        lu = x[:, :3]
        fea = self.embedding(x)
        illu = self.illu_proj(illu_fea)
        if illu.shape[-2:] != fea.shape[-2:]:
            illu = F.interpolate(illu, size=fea.shape[-2:], mode="bilinear", align_corners=False)

        fea_encoder = []
        illu_list = []
        for igab, fea_down, illu_down in self.encoder_layers:
            fea = igab(fea, illu)
            fea_encoder.append(fea)
            illu_list.append(illu)
            fea = fea_down(fea)
            illu = illu_down(illu)

        fea = self.bottleneck(fea, illu)

        for i, (up, fuse, igab) in enumerate(self.decoder_layers):
            skip = fea_encoder[self.level - 1 - i]
            fea = self._align(up(fea), skip)
            fea = fuse(torch.cat([fea, skip], dim=1))
            illu = illu_list[self.level - 1 - i]
            fea = igab(fea, illu)

        return lu + self.mapping(fea)


class FrequencyRestorationBranch(nn.Module):
    """Amplitude correction (+ light phase refine) in the frequency domain.

    Identity-safe at init: amp gate ≈ 1 and spatial residual ≈ 0, so output ≈ input.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.amp_net = nn.Sequential(
            conv1x1(3, channels),
            nn.GELU(),
            conv3x3(channels, channels),
            nn.GELU(),
            ResBlock(channels),
            conv1x1(channels, 3),
        )
        self.refine_in = nn.Sequential(conv3x3(6, channels), nn.GELU())
        self.refine_ig = IllumGuidedBlock(channels)
        self.refine_out = nn.Sequential(ResBlock(channels), ResBlock(channels), conv3x3(channels, 3))
        # gate = 1 + tanh(0) => 1; refine residual = 0
        nn.init.zeros_(self.amp_net[-1].weight)
        nn.init.zeros_(self.amp_net[-1].bias)
        nn.init.zeros_(self.refine_out[-1].weight)
        nn.init.zeros_(self.refine_out[-1].bias)

    def forward(self, lu: torch.Tensor, l_map: torch.Tensor) -> torch.Tensor:
        luc = lu.clamp(0.0, 1.5)
        fft = torch.fft.rfft2(luc, norm="ortho")
        amp = torch.abs(fft).clamp_min(1e-6)
        phase = torch.angle(fft)
        # Multiplicative amp gate centered at 1 (identity) instead of softplus+0.5.
        gate = 1.0 + torch.tanh(self.amp_net(torch.log(amp)))
        amp2 = amp * gate
        rec = torch.fft.irfft2(
            torch.complex(amp2 * torch.cos(phase), amp2 * torch.sin(phase)),
            s=luc.shape[-2:],
            norm="ortho",
        )
        h = self.refine_in(torch.cat([luc, rec.clamp(0.0, 1.5)], dim=1))
        h = self.refine_ig(h, l_map)
        h = self.refine_out(h)
        return lu + h


class NoiseAwareFusion(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(7, channels),
            nn.GELU(),
            conv3x3(channels, channels),
            nn.GELU(),
            ResBlock(channels),
            conv1x1(channels, 1),
            # raw logit -> sigmoid outside so we can bias toward spatial branch at init
        )
        self.mix_in = nn.Sequential(conv3x3(7, channels), nn.GELU())
        self.mix_res = ResBlock(channels)
        self.mix_out = conv3x3(channels, 3)
        # Prefer r_spa at init (m≈1) and zero residual so fused ≈ r_spa.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)
        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.mix_out.bias)

    def forward(
        self,
        i: torch.Tensor,
        l_hat: torch.Tensor,
        r_spa: torch.Tensor,
        r_fre: torch.Tensor,
        lu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gray = i.mean(dim=1, keepdim=True)
        gx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        grad = torch.sqrt(gx * gx + gy * gy + 1e-8)
        mu = F.avg_pool2d(gray, 7, 1, 3)
        mu2 = F.avg_pool2d(gray * gray, 7, 1, 3)
        local_std = torch.sqrt((mu2 - mu * mu).clamp_min(0) + 1e-8)
        dark = 1.0 - l_hat
        diff = (r_spa - r_fre).abs().mean(1, keepdim=True)
        feat = torch.cat([grad, local_std, dark, gray, l_hat, diff, lu.mean(1, keepdim=True)], dim=1)
        m = torch.sigmoid(self.net(feat))
        fused = m * r_spa + (1.0 - m) * r_fre
        h = self.mix_out(self.mix_res(self.mix_in(torch.cat([fused, lu.clamp(0.0, 1.5), l_hat], dim=1))))
        return fused + h, m


class FSNANet(nn.Module):
    def __init__(
        self,
        base: int = 40,
        use_srb: bool = True,
        use_frb: bool = True,
        use_naf: bool = True,
        use_ecm: bool = False,
        igab_blocks: tuple[int, ...] = (1, 2, 2),
    ):
        super().__init__()
        self.use_srb = use_srb
        self.use_frb = use_frb
        self.use_naf = use_naf
        self.use_ecm = bool(use_ecm)  # kept for CLI compat; IGAB family ignores gain head
        self.ie = IlluminationEstimator(base)
        self.srb = (
            SpatialRestorationBranch(7, base, level=2, num_blocks=igab_blocks) if use_srb else None
        )
        self.frb = FrequencyRestorationBranch(base) if use_frb else None
        self.naf = NoiseAwareFusion(max(base // 2, 16)) if use_naf else None
        self.out_in = nn.Sequential(conv3x3(7, base), nn.GELU())
        self.out_ig = IllumGuidedBlock(base)
        self.out_res = nn.Sequential(ResBlock(base), ResBlock(base))
        self.out_rgb = conv3x3(base, 3)
        # IGAB family: no FullImageGainHead / GEC / ECM.
        self.fige = None
        self.gec = None

    def forward(self, i: torch.Tensor) -> dict[str, torch.Tensor]:
        e_map, l_hat, illu_fea = self.ie(i)
        lu = i * e_map
        lu_in = lu.clamp(0.0, 1.5)

        r_spa = (
            self.srb(torch.cat([lu_in, i, l_hat], dim=1), illu_fea)
            if self.srb is not None
            else lu
        )
        r_fre = self.frb(lu, l_hat) if self.frb is not None else lu

        if self.naf is not None and self.srb is not None and self.frb is not None:
            fused, m = self.naf(i, l_hat, r_spa, r_fre, lu_in)
        elif self.srb is not None and self.frb is not None:
            fused, m = 0.5 * (r_spa + r_fre), torch.full_like(l_hat, 0.5)
        elif self.srb is not None:
            fused, m = r_spa, torch.ones_like(l_hat)
        elif self.frb is not None:
            fused, m = r_fre, torch.zeros_like(l_hat)
        else:
            fused, m = lu, torch.zeros_like(l_hat)

        h = self.out_in(torch.cat([fused, lu_in, l_hat], dim=1))
        h = self.out_res(self.out_ig(h, l_hat))
        i_hat = (fused + self.out_rgb(h)).clamp(0.0, 1.0)
        gain = torch.ones(i_hat.shape[0], 3, 1, 1, device=i_hat.device, dtype=i_hat.dtype)
        bias = torch.zeros_like(gain)

        return {
            "enhanced": i_hat,
            "enhanced_raw": i_hat,
            "illumination": l_hat,
            "illumination_adj": l_hat,
            "illu_fea": illu_fea,
            "reflectance": lu,
            "mask": m,
            "r_spa": r_spa,
            "r_fre": r_fre,
            "lightup": lu,
            "lightup_map": e_map,
            "fused": fused,
            "expo_gain": gain,
            "expo_bias": bias,
        }


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    net = FSNANet()
    x = torch.randn(1, 3, 128, 128)
    y = net(x)
    n = count_parameters(net)
    print(y["enhanced"].shape, f"{n:.3f}M", "fige", net.fige is not None)
    assert 1.5 <= n <= 6.0, f"params {n:.3f}M out of range"
