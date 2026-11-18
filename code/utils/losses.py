"""Training losses for FSNA-Net (v9 clean exposure)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return 1.0 - ssim_map.clamp(0, 1).mean()


class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def norm(x):
            return (x - self.mean) / self.std

        return F.l1_loss(self.vgg(norm(pred)), self.vgg(norm(target)))


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def multiscale_charbonnier(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = charbonnier(pred, target)
    for scale in (2, 4):
        p = F.avg_pool2d(pred, scale)
        t = F.avg_pool2d(target, scale)
        loss = loss + 0.5 * charbonnier(p, t)
    return loss


class FSNALoss(nn.Module):
    """v9: blueprint main losses + clean global-mean exposure only.

    Disabled competitors that fought the gain head in v6–v8:
    fuse / branch / raw_expo preemption. Exposure supervises final Î means only.
    """

    def __init__(
        self,
        w_l1=1.0,
        w_ssim=0.5,
        w_perc=0.1,
        w_noise=0.5,
        w_tv=0.01,
        w_fuse=0.0,
        w_expo=0.5,
        w_branch=0.0,
        use_perc=True,
        use_noise=True,
        expo_only: bool = False,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perc = w_perc
        self.w_noise = w_noise
        self.w_tv = w_tv
        self.w_fuse = w_fuse
        self.w_expo = w_expo
        self.w_branch = w_branch
        self.use_perc = use_perc
        self.use_noise = use_noise
        # Full-image step: emphasize L1 + expo (texture already covered by patch step).
        self.expo_only = expo_only
        self.perc = VGGPerceptual() if use_perc else None

    def forward(self, out: dict, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pred = out["enhanced"]
        l1 = F.l1_loss(pred, target) + 0.5 * multiscale_charbonnier(pred, target)
        if self.expo_only:
            ssim = pred.new_tensor(0.0)
            perc = pred.new_tensor(0.0)
            noise = pred.new_tensor(0.0)
            tv = pred.new_tensor(0.0)
            fuse_loss = pred.new_tensor(0.0)
            branch = pred.new_tensor(0.0)
        else:
            ssim = ssim_loss(pred, target)
            perc = self.perc(pred, target) if self.perc is not None else pred.new_tensor(0.0)
            mask = out.get("mask")
            if self.use_noise and mask is not None:
                # Detach M: without detach, ||M*(pred-gt)|| pushes M→0 and kills NAF.
                dark = (1.0 - out["illumination"]).clamp(0.0, 1.0).detach()
                noise = (mask.detach() * (1.0 + dark) * (pred - target).abs()).mean()
            else:
                noise = pred.new_tensor(0.0)
            tv = tv_loss(out["illumination"])
            fused = out.get("fused")
            fuse_loss = (
                charbonnier(fused.clamp(0.0, 1.0), target)
                if (fused is not None and self.w_fuse > 0)
                else pred.new_tensor(0.0)
            )
            # v9: branch supervision off by default (w_branch=0) — it preempted gain.
            spa = out.get("r_spa")
            fre = out.get("r_fre")
            branch = pred.new_tensor(0.0)
            if self.w_branch > 0:
                if spa is not None:
                    branch = branch + 0.15 * charbonnier(spa.clamp(0.0, 1.0), target)
                if fre is not None:
                    branch = branch + 0.10 * charbonnier(fre.clamp(0.0, 1.0), target)
                branch = self.w_branch * branch

        # Exposure: final Î global means only (no raw_expo / local pools / hard).
        pred_mean = pred.mean(dim=(2, 3), keepdim=True)
        tgt_mean = target.mean(dim=(2, 3), keepdim=True)
        ch_mean = (pred_mean - tgt_mean).abs().mean()
        ch_mse = F.mse_loss(pred_mean, tgt_mean)
        lum_pred = pred.mean(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
        lum_tgt = target.mean(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
        lum = (lum_pred - lum_tgt).abs().mean()
        expo = 2.0 * ch_mean + 1.5 * ch_mse + 1.0 * lum

        if self.expo_only:
            # Full-image step: L1 + stronger expo; skip heavy perceptual/branch terms.
            total = self.w_l1 * l1 + self.w_expo * expo
        else:
            total = (
                self.w_l1 * l1
                + self.w_ssim * ssim
                + self.w_perc * perc
                + self.w_noise * noise
                + self.w_tv * tv
                + self.w_fuse * fuse_loss
                + branch
                + self.w_expo * expo
            )
        logs = {
            "l1": float(l1.detach()),
            "ssim": float(ssim.detach()) if torch.is_tensor(ssim) else float(ssim),
            "perc": float(perc.detach()) if torch.is_tensor(perc) else float(perc),
            "noise": float(noise.detach()) if torch.is_tensor(noise) else float(noise),
            "tv": float(tv.detach()) if torch.is_tensor(tv) else float(tv),
            "fuse": float(fuse_loss.detach()) if torch.is_tensor(fuse_loss) else float(fuse_loss),
            "branch": float(branch.detach()) if torch.is_tensor(branch) else float(branch),
            "expo": float(expo.detach()),
            "total": float(total.detach()),
        }
        return total, logs
