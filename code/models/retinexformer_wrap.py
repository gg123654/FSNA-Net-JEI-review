"""Official RetinexFormer + FSNA-on-RF (Route A).

Loads vendor arch from third_party/Retinexformer via importlib (avoids basicsr
package deps like lmdb). Recipe: n_feat=40, stage=1, blocks=[1,2,2].
FSNA-RF keeps that backbone and adds FRB + NAF residual fusion.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "Retinexformer"
_ARCH = _VENDOR / "basicsr" / "models" / "archs" / "RetinexFormer_arch.py"


def _load_retinexformer_cls():
    if not _ARCH.is_file():
        raise FileNotFoundError(f"Missing official arch: {_ARCH}")
    # Ensure sibling imports inside the arch file can resolve einops etc.
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    spec = importlib.util.spec_from_file_location("retinexformer_arch_vendor", _ARCH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_ARCH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RetinexFormer


RetinexFormer = _load_retinexformer_cls()

from .fsna_net import FrequencyRestorationBranch, NoiseAwareFusion  # noqa: E402


def build_retinexformer(n_feat: int = 40, stage: int = 1, num_blocks=(1, 2, 2)) -> RetinexFormer:
    return RetinexFormer(
        in_channels=3,
        out_channels=3,
        n_feat=n_feat,
        stage=stage,
        num_blocks=list(num_blocks),
    )


class FSNARFNet(nn.Module):
    """Frozen-friendly RetinexFormer + FRB/NAF residual corrector.

    Identity at init: FRB is identity-safe, NAF starts near 50/50 with zero mix
    residual, fuse gate starts mid with a floor, res_head is zero-init — so
    enhanced ≈ backbone (~25.16 PSNR). Main path goes through fused (not base+δ
    only), so FRB/NAF gradients are not bypassed.
    """

    def __init__(
        self,
        n_feat: int = 40,
        stage: int = 1,
        num_blocks=(1, 2, 2),
        use_frb: bool = True,
        use_naf: bool = True,
        freeze_backbone: bool = False,
        res_amp: float = 0.25,
        gate_floor: float = 0.15,
        mask_floor: float = 0.0,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.backbone = build_retinexformer(n_feat=n_feat, stage=stage, num_blocks=num_blocks)
        self.use_frb = use_frb
        self.use_naf = use_naf
        self.res_amp = float(res_amp)
        self.gate_floor = float(gate_floor)
        self.mask_floor = float(mask_floor)
        self.frb = FrequencyRestorationBranch(max(n_feat // 2, 16)) if use_frb else None
        self.naf = NoiseAwareFusion(max(n_feat // 2, 16)) if use_naf else None
        # Default NAF bias=4 locks mask≈1 (ignore FRB). For RF residual corrector
        # start near 50/50 so the frequency branch is not dead at init.
        if self.naf is not None:
            nn.init.zeros_(self.naf.net[-1].bias)
        self.l_est = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        # Scalar fuse gate: enhanced path must keep a non-zero share of NAF fused.
        # Init logit=0 → sigmoid=0.5; floor prevents collapse to pure backbone.
        self.fuse_logit = nn.Parameter(torch.zeros(1))
        # Residual corrector on top of fused; last layer zero => delta=0 at init.
        # Imaging rationale: residual amplitude is modulated by darkness / local
        # noise cues so correction focuses where electronic imaging noise grows.
        self.res_head = nn.Sequential(
            nn.Conv2d(13, 48, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(48, 3, 3, 1, 1),
        )
        self.res_amp_map = nn.Sequential(
            nn.Conv2d(5, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.res_head[-1].weight)
        nn.init.zeros_(self.res_head[-1].bias)
        # Start amp_map near ~0.5 so early residual stays small.
        nn.init.zeros_(self.res_amp_map[-2].weight)
        nn.init.zeros_(self.res_amp_map[-2].bias)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Optional: thaw only the last N leaf Parameter tensors (tiny soft-FT).
            if unfreeze_last_n > 0:
                bb_params = [p for p in self.backbone.parameters()]
                for p in bb_params[-int(unfreeze_last_n) :]:
                    p.requires_grad = True

    def forward(self, i: torch.Tensor) -> dict[str, torch.Tensor]:
        # Detach base when frozen so residual/aux grads cannot touch RF weights
        # even if freeze flags were missed somewhere.
        base_live = self.backbone(i).clamp(0.0, 1.0)
        base = base_live.detach() if not any(p.requires_grad for p in self.backbone.parameters()) else base_live
        l_hat = self.l_est(i)
        r_fre = self.frb(base, l_hat) if self.frb is not None else base
        if self.naf is not None and self.frb is not None:
            naf_out, m = self.naf(i, l_hat, base, r_fre, base)
            if self.mask_floor > 0:
                # Soft floor on the logged/amp mask so noise/amp cues cannot die at 0.
                # NAF output itself stays as trained; floor is a training stabilizer signal.
                m = self.mask_floor + (1.0 - self.mask_floor) * m
        else:
            naf_out, m = base, torch.full_like(l_hat, 0.5)
        # Anti-collapse blend: gate ∈ [floor, 1], init ≈ 0.5*(1+floor)/(≈0.575 if floor=0.15)
        g = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(self.fuse_logit)
        fused = (1.0 - g) * base + g * naf_out
        feat = torch.cat([i, base, fused, r_fre, l_hat], dim=1)
        gray = i.mean(dim=1, keepdim=True)
        # Local std as a cheap noise/texture cue (electronic imaging).
        mu = F.avg_pool2d(gray, 5, 1, 2)
        var = (F.avg_pool2d(gray * gray, 5, 1, 2) - mu * mu).clamp_min(0.0).sqrt()
        amp_in = torch.cat([gray, 1.0 - l_hat, var, m, fused.mean(dim=1, keepdim=True)], dim=1)
        amp = 0.35 + 0.65 * self.res_amp_map(amp_in)
        delta = (self.res_amp * amp) * torch.tanh(self.res_head(feat))
        # Main path uses fused so FRB/NAF contribute; identity-safe at init.
        enhanced = (fused + delta).clamp(0.0, 1.0)
        gate_map = g.view(1, 1, 1, 1).expand_as(m)
        return {
            "enhanced": enhanced,
            "base": base_live.clamp(0.0, 1.0),
            "r_spa": base,
            "r_fre": r_fre if self.frb is not None else base,
            "fused": fused.clamp(0.0, 1.0),
            "delta": delta,
            "mask": m,
            "gate": gate_map,
            "l_hat": l_hat,
            "light_up": base_live,
            "res_amp_map": amp,
        }


def count_rf_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
