"""Evaluation metrics: PSNR / SSIM / LPIPS / NIQE / BRISQUE."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    # pred/target: 1x3xHxW or 3xHxW
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 11, 1, 5)
    mu_y = F.avg_pool2d(target, 11, 1, 5)
    sigma_x = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return float(ssim_map.mean().item())


class LPIPSMetric:
    def __init__(self, device="cuda"):
        import lpips

        self.model = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred = pred.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1
        return float(self.model(pred, target).mean().item())


def niqe_brisque_pair(img_rgb_u8: np.ndarray) -> tuple[float, float]:
    """Compute NIQE/BRISQUE if pyiqa available; else return NaN."""
    try:
        import pyiqa
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.from_numpy(img_rgb_u8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = t.to(device)
        if not hasattr(niqe_brisque_pair, "_niqe"):
            niqe_brisque_pair._niqe = pyiqa.create_metric("niqe", device=device)
            niqe_brisque_pair._brisque = pyiqa.create_metric("brisque", device=device)
        niqe = float(niqe_brisque_pair._niqe(t).item())
        brisque = float(niqe_brisque_pair._brisque(t).item())
        return niqe, brisque
    except Exception:
        # Fallback: simple no-reference proxies based on sharpness/contrast (not used as claim)
        gray = img_rgb_u8.mean(axis=2).astype(np.float32)
        contrast = float(gray.std())
        # Return placeholders that will be overwritten if pyiqa installs
        return float("nan"), float("nan")
