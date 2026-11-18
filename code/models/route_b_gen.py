"""Route B: RF-GEN — frozen RetinexFormer + learned Global Exposure Network.

Rationale (from results/route_b_error_diagnosis_lolv1.json):
RF's dominant error on LOL-v1 is GLOBAL exposure. A per-image global affine
oracle lifts 25.16 -> 28.44 (+3.28 dB); gamma oracle -> 27.23. RF systematically
under-exposes on GT-brighter images. Hand scalar regressors failed (R^2<0), so we
use a LEARNED whole-image context encoder predicting a multiplicative
gamma/gain/bias with the correct inductive bias, plus a tiny local detail net.

Identity-safe at init: gain=1, gamma=1, bias=0, detail=0  => enhanced == base (RF).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retinexformer_wrap import build_retinexformer


class GlobalExposureNet(nn.Module):
    """Downsample whole image -> GAP -> MLP -> per-channel gain/gamma/bias."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, ch, 5, 2, 2), nn.GELU(),
            nn.Conv2d(ch, ch, 5, 2, 2), nn.GELU(),
            nn.Conv2d(ch, ch, 5, 2, 2), nn.GELU(),
            nn.Conv2d(ch, ch, 5, 2, 2), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(ch * 2, ch), nn.GELU(), nn.Linear(ch, 9),
        )
        # start at identity
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, low: torch.Tensor, base: torch.Tensor):
        x = torch.cat([low, base], dim=1)
        h = self.stem(x)
        # global average + global max pool (both whole-image context)
        g = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        p = self.head(g)
        gain = 1.0 + 0.35 * torch.tanh(p[:, 0:3])
        gamma = 1.0 + 0.20 * torch.tanh(p[:, 3:6])
        bias = 0.08 * torch.tanh(p[:, 6:9])
        return gain, gamma, bias


class RFGENNet(nn.Module):
    def __init__(self, n_feat: int = 40, stage: int = 1, num_blocks=(1, 2, 2), ge_ch: int = 32, detail_amp: float = 0.15):
        super().__init__()
        self.backbone = build_retinexformer(n_feat, stage, num_blocks)
        self.ge = GlobalExposureNet(ge_ch)
        self.detail_amp = float(detail_amp)
        self.local = nn.Sequential(
            nn.Conv2d(6, ge_ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ge_ch, ge_ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ge_ch, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # backbone stays eval (frozen, BN-free but keep deterministic)
        self.backbone.eval()
        return self

    def _base(self, low: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(low).clamp(0.0, 1.0)

    def forward(self, low: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self._base(low)
        b = base.shape[0]
        gain, gamma, bias = self.ge(low, base)
        gain = gain.view(b, 3, 1, 1)
        gamma = gamma.view(b, 3, 1, 1)
        bias = bias.view(b, 3, 1, 1)
        corr = (base.clamp(1e-4, 1.0).pow(gamma) * gain + bias).clamp(0.0, 1.0)
        detail = self.detail_amp * torch.tanh(self.local(torch.cat([low, corr], dim=1)))
        enhanced = (corr + detail).clamp(0.0, 1.0)
        return {
            "enhanced": enhanced,
            "base": base,
            "gain": gain.view(b, 3),
            "gamma": gamma.view(b, 3),
            "bias": bias.view(b, 3),
            "detail": detail,
        }


class FSNAGENNet(RFGENNet):
    """Same learned global-exposure head, but base = frozen FSNA (FSNARFNet) output.

    FSNA keeps better SSIM/LPIPS (0.872 / 0.1249) than RF; adding the learned
    per-image exposure head on top should lift PSNR (~25.6) while preserving those.
    """

    def __init__(self, fsna: nn.Module, ge_ch: int = 32, detail_amp: float = 0.15):
        nn.Module.__init__(self)
        self.backbone = fsna  # FSNARFNet, returns dict; frozen
        self.ge = GlobalExposureNet(ge_ch)
        self.detail_amp = float(detail_amp)
        self.local = nn.Sequential(
            nn.Conv2d(6, ge_ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ge_ch, ge_ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ge_ch, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def _base(self, low: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(low)
        return (out["enhanced"] if isinstance(out, dict) else out).clamp(0.0, 1.0)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
