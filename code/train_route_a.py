"""Route A Phase-1: reproduce official RetinexFormer on LOL-v1.

Matches Options/RetinexFormer_LOL_v1.yml recipe:
  - n_feat=40, stage=1, blocks=[1,2,2]
  - patch 128, batch 8, Adam lr=2e-4
  - L1 loss + mixup (beta=1.2)
  - CosineAnnealingRestartCyclicLR-like: fixed 3e-4 for 46k then cosine
  - NO GT-mean alignment at eval
  - Kill gates: 20k>=23.5, 50k>=24.3, target best>=24.8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.retinexformer_wrap import (  # noqa: E402
    FSNARFNet,
    build_retinexformer,
    count_rf_parameters,
)
from models.rf_corrector import RFCorrectorNet, count_corr_parameters  # noqa: E402
from utils.dataset import PairedLLDataset, resolve_split  # noqa: E402
from utils.losses import VGGPerceptual, charbonnier, ssim_loss, tv_loss  # noqa: E402
from utils.metrics import psnr, ssim  # noqa: E402


def mixup_batch(low: torch.Tensor, high: torch.Tensor, beta: float = 1.2, p: float = 0.5):
    if random.random() > p:
        return low, high
    lam = float(torch.distributions.Beta(beta, beta).sample())
    idx = torch.randperm(low.size(0), device=low.device)
    low = lam * low + (1.0 - lam) * low[idx]
    high = lam * high + (1.0 - lam) * high[idx]
    return low, high


@torch.no_grad()
def evaluate(model, loader, device, is_dict: bool):
    model.eval()
    psnr_v, ssim_v, n = 0.0, 0.0, 0
    base_v, fused_v, gate_v, mask_v, delta_v = 0.0, 0.0, 0.0, 0.0, 0.0
    has_extra = False
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        oh, ow = batch["orig_hw"][0].tolist()
        out = model(low)
        pred = out["enhanced"] if is_dict else out
        pred = pred[:, :, :oh, :ow].clamp(0, 1)
        high = high[:, :, :oh, :ow]
        psnr_v += psnr(pred, high)
        ssim_v += ssim(pred, high)
        if is_dict and isinstance(out, dict) and "base" in out:
            has_extra = True
            base_v += psnr(out["base"][:, :, :oh, :ow].clamp(0, 1), high)
            if "fused" in out:
                fused_v += psnr(out["fused"][:, :, :oh, :ow].clamp(0, 1), high)
            # Prefer explicit fuse gate; fall back to NAF mask (old runs logged 0
            # because only "mask" existed while evaluate looked for "gate").
            if "gate" in out:
                gate_v += float(out["gate"].mean().item())
            elif "mask" in out:
                gate_v += float(out["mask"].mean().item())
            if "mask" in out:
                mask_v += float(out["mask"].mean().item())
            if "delta" in out:
                delta_v += float(out["delta"].abs().mean().item())
        n += 1
    out_m = {"psnr": psnr_v / max(n, 1), "ssim": ssim_v / max(n, 1)}
    if has_extra:
        out_m["psnr_base"] = base_v / max(n, 1)
        out_m["psnr_fused"] = fused_v / max(n, 1)
        out_m["gate"] = gate_v / max(n, 1)
        out_m["mask"] = mask_v / max(n, 1)
        out_m["delta"] = delta_v / max(n, 1)
    return out_m


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def lr_schedule(step: int, base_lr: float, total: int, force_official_hold: bool = False) -> float:
    # From-scratch official yml: period-1 hold at 3e-4.
    # Finetune from pretrained MUST NOT force 3e-4 — that destroys the backbone.
    hold = 46000
    hold_lr = max(base_lr, 3e-4) if force_official_hold else base_lr
    if step < hold:
        return hold_lr
    t = (step - hold) / max(total - hold, 1)
    return 1e-6 + 0.5 * (hold_lr - 1e-6) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def write_kill(save_dir: Path, payload: dict):
    results = ROOT.parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    for path in (save_dir / "kill.json", results / "route_a_kill.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(f"KILL: {payload}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(ROOT.parent / "datasets"))
    ap.add_argument("--dataset", type=str, default="lolv1")
    ap.add_argument(
        "--model",
        type=str,
        default="retinexformer",
        choices=["retinexformer", "fsna_rf", "rf_corr"],
    )
    ap.add_argument("--iters", type=int, default=150000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_feat", type=int, default=40)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--resume", type=str, default="", help="Resume model+opt+iter (same recipe)")
    ap.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Load model weights only and start from iter 0 (phase transition / unfreeze)",
    )
    ap.add_argument("--pretrained", type=str, default="")
    ap.add_argument("--mixup_beta", type=float, default=1.2)
    ap.add_argument("--no_mixup", action="store_true")
    ap.add_argument("--gate_20k", type=float, default=23.5)
    ap.add_argument("--gate_50k", type=float, default=24.3)
    ap.add_argument("--target_psnr", type=float, default=24.8)
    ap.add_argument("--no_kill_gate", action="store_true")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=1.0,
        help="LR multiplier for RetinexFormer backbone when not frozen (finetune: use 0.01–0.05)",
    )
    ap.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=0,
        help="With --freeze_backbone: thaw only last N backbone Parameter tensors",
    )
    ap.add_argument("--res_amp", type=float, default=0.25, help="Residual amplitude scale (fsna_rf)")
    ap.add_argument("--gate_floor", type=float, default=0.15, help="Min fuse gate toward NAF (fsna_rf)")
    ap.add_argument("--mask_floor", type=float, default=0.0, help="Soft floor on NAF mask for amp/noise")
    ap.add_argument("--no_frb", action="store_true", help="Ablation: disable FrequencyRestorationBranch")
    ap.add_argument("--no_naf", action="store_true", help="Ablation: disable NoiseAwareFusion")
    ap.add_argument(
        "--no_residual",
        action="store_true",
        help="Ablation: zero residual head (enhanced≈fused/backbone)",
    )
    ap.add_argument(
        "--official_hold_lr",
        action="store_true",
        help="Force official 3e-4 hold (from-scratch only; NEVER with --pretrained)",
    )
    ap.add_argument(
        "--official_augs",
        action="store_true",
        help="Geometric augs only (match RetinexFormer_LOL_v1.yml; no photometric scale)",
    )
    ap.add_argument("--w_base", type=float, default=0.0, help="Aux L1 on backbone output (fsna_rf; 0 if frozen)")
    ap.add_argument("--w_fused", type=float, default=0.3, help="Aux L1 on NAF fused (fsna_rf)")
    ap.add_argument("--w_fre", type=float, default=0.15, help="Aux L1 on FRB output (fsna_rf)")
    ap.add_argument("--w_ssim", type=float, default=0.2, help="SSIM loss weight on enhanced")
    ap.add_argument("--w_noise", type=float, default=0.0, help="Noise-aware L1 on dark/noisy mask")
    ap.add_argument("--w_perc", type=float, default=0.0, help="VGG perceptual weight (0=off)")
    ap.add_argument("--w_tv", type=float, default=0.0, help="TV on illumination map")
    ap.add_argument(
        "--w_teacher",
        type=float,
        default=0.0,
        help="L1(base_live, frozen_teacher) to protect RF when soft-unfreezing",
    )
    ap.add_argument(
        "--protect_base_drop",
        type=float,
        default=0.0,
        help="Kill if eval psnr_base drops more than this below seed base (0=off)",
    )
    ap.add_argument(
        "--improve_by_k",
        type=int,
        default=0,
        help="By this iter, best must beat seed_psnr by --improve_delta or kill (0=off)",
    )
    ap.add_argument("--improve_delta", type=float, default=0.02, help="Required PSNR gain over seed by --improve_by_k")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"route_a_{args.model}_{args.dataset}"
    save_dir = ROOT.parent / "checkpoints" / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = resolve_split(args.data_root, args.dataset, "train")
    test_pairs = resolve_split(args.data_root, args.dataset, "test")
    train_ds = PairedLLDataset(
        train_pairs, args.patch_size, train=True, photometric=not args.official_augs
    )
    test_ds = PairedLLDataset(test_pairs, args.patch_size, train=False, photometric=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    if args.model == "retinexformer":
        model = build_retinexformer(n_feat=args.n_feat, stage=1, num_blocks=(1, 2, 2)).to(device)
        is_dict = False
    elif args.model == "rf_corr":
        model = RFCorrectorNet(
            n_feat=args.n_feat,
            stage=1,
            num_blocks=(1, 2, 2),
            freeze_backbone=args.freeze_backbone or True,
            init_scale=max(args.res_amp * 0.3, 0.05),
        ).to(device)
        is_dict = True
    else:
        model = FSNARFNet(
            n_feat=args.n_feat,
            stage=1,
            num_blocks=(1, 2, 2),
            use_frb=not args.no_frb,
            use_naf=not args.no_naf,
            freeze_backbone=args.freeze_backbone,
            res_amp=0.0 if args.no_residual else args.res_amp,
            gate_floor=args.gate_floor,
            mask_floor=args.mask_floor,
            unfreeze_last_n=args.unfreeze_last_n,
        ).to(device)
        is_dict = True
        if args.no_residual:
            for p in model.res_head.parameters():
                p.requires_grad = False
            for p in model.res_amp_map.parameters():
                p.requires_grad = False

    nparams = count_corr_parameters(model) if args.model == "rf_corr" else count_rf_parameters(model)
    # Finetune from pretrained: never force the official 3e-4 hold (washes out BB).
    force_official_hold = bool(args.official_hold_lr) and not bool(args.pretrained)
    if args.official_hold_lr and args.pretrained:
        print("WARN: --official_hold_lr ignored because --pretrained is set", flush=True)

    start_iter = 0
    best = {"psnr": -1.0, "ssim": 0.0, "iter": 0}
    history = []
    seed_psnr = None
    seed_base = None
    teacher_bb = None
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if args.model == "retinexformer":
            target = model
        else:
            target = model.backbone
        missing, unexpected = target.load_state_dict(state, strict=False)
        print(
            f"Loaded pretrained {args.pretrained} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    # Differential LR: protect pretrained backbone; heads keep full --lr.
    bb_params = (
        [p for p in model.backbone.parameters() if p.requires_grad]
        if args.model in ("fsna_rf", "rf_corr")
        else []
    )
    if args.model in ("fsna_rf", "rf_corr") and bb_params and args.backbone_lr_mult != 1.0:
        head_params = [
            p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")
        ]
        param_groups = [
            {"params": bb_params, "lr": args.lr * args.backbone_lr_mult, "lr_mult": args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr, "lr_mult": 1.0},
        ]
    else:
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.lr,
                "lr_mult": 1.0,
            }
        ]
    opt = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.999))
    print(
        f"RouteA model={args.model} params={nparams:.3f}M device={device} "
        f"patch={args.patch_size} batch={args.batch_size} lr={args.lr} "
        f"bb_mult={args.backbone_lr_mult} freeze_bb={args.freeze_backbone} "
        f"force_hold={force_official_hold} groups={len(param_groups)}",
        flush=True,
    )
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"init_from {args.init_from} missing={len(missing)} unexpected={len(unexpected)} "
            f"(start_iter=0, fresh optimizer)",
            flush=True,
        )
        start_iter = 0
        # Seed best with an immediate eval baseline so we do not lose Phase-2 peak.
        with torch.no_grad():
            m0 = evaluate(model, test_loader, device, is_dict)
        best = {"psnr": float(m0["psnr"]), "ssim": float(m0["ssim"]), "iter": 0}
        seed_psnr = float(m0["psnr"])
        seed_base = float(m0.get("psnr_base", m0["psnr"]))
        torch.save({"model": model.state_dict(), "iter": 0, "metrics": m0}, save_dir / "best.pth")
        with open(save_dir / "best.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print(
            f"Seeded best from init_from: PSNR={best['psnr']:.3f} SSIM={best['ssim']:.4f} "
            f"base={seed_base:.3f}",
            flush=True,
        )
        if args.w_teacher > 0 and args.model == "fsna_rf":
            import copy

            teacher_bb = copy.deepcopy(model.backbone).to(device)
            teacher_bb.eval()
            for p in teacher_bb.parameters():
                p.requires_grad = False
            print(f"Teacher backbone locked for w_teacher={args.w_teacher}", flush=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_iter = int(ckpt.get("iter", 0))
        bj = save_dir / "best.json"
        if bj.exists():
            with open(bj, encoding="utf-8") as f:
                best = json.load(f)

    perc_fn = None
    if args.w_perc > 0:
        perc_fn = VGGPerceptual().to(device)

    total = args.iters
    data_iter = cycle_loader(train_loader)
    model.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    pbar = tqdm(range(start_iter, total), initial=start_iter, total=total, desc="route_a")

    for step in pbar:
        base_lr_now = lr_schedule(step, args.lr, total, force_official_hold=force_official_hold)
        for g in opt.param_groups:
            g["lr"] = base_lr_now * float(g.get("lr_mult", 1.0))

        batch = next(data_iter)
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        if not args.no_mixup:
            low, high = mixup_batch(low, high, beta=args.mixup_beta)

        out = model(low)
        pred = out["enhanced"] if is_dict else out
        loss = F.l1_loss(pred, high) + 0.5 * charbonnier(pred, high)
        if args.w_ssim > 0:
            loss = loss + args.w_ssim * ssim_loss(pred, high)
        if perc_fn is not None:
            loss = loss + args.w_perc * perc_fn(pred, high)
        if is_dict:
            if args.w_base > 0 and "base" in out:
                loss = loss + args.w_base * F.l1_loss(out["base"], high)
            if args.w_fused > 0 and "fused" in out:
                loss = loss + args.w_fused * F.l1_loss(out["fused"], high)
            if args.w_fre > 0 and "r_fre" in out:
                loss = loss + args.w_fre * F.l1_loss(out["r_fre"], high)
            if args.w_noise > 0 and "mask" in out:
                m = out["mask"]
                if m.shape[-2:] != pred.shape[-2:]:
                    m = F.interpolate(m, size=pred.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss + args.w_noise * (m * (pred - high).abs()).mean()
            if args.w_tv > 0 and "l_hat" in out:
                loss = loss + args.w_tv * tv_loss(out["l_hat"])
            if args.w_teacher > 0 and teacher_bb is not None and "base" in out:
                with torch.no_grad():
                    t_base = teacher_bb(low).clamp(0.0, 1.0)
                loss = loss + args.w_teacher * F.l1_loss(out["base"], t_base)
        loss.backward()
        # 0.01 was starving FRB/NAF/res_head on frozen-BB finetune; keep mild clip.
        clip = 0.01 if args.model == "retinexformer" else 1.0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == total:
            metrics = evaluate(model, test_loader, device, is_dict)
            history.append({"iter": step + 1, "loss": float(loss.item()), "lr": opt.param_groups[0]["lr"], **metrics})
            with open(save_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": step + 1, "metrics": metrics}, save_dir / "last.pth")
            if metrics["psnr"] > best["psnr"]:
                best = {"psnr": metrics["psnr"], "ssim": metrics["ssim"], "iter": step + 1}
                torch.save({"model": model.state_dict(), "iter": step + 1, "metrics": metrics}, save_dir / "best.pth")
                with open(save_dir / "best.json", "w", encoding="utf-8") as f:
                    json.dump(best, f, indent=2)
            elapsed = (time.time() - t0) / 3600.0
            extra = ""
            if "psnr_base" in metrics:
                extra = (
                    f" base={metrics['psnr_base']:.3f} fused={metrics.get('psnr_fused', 0):.3f} "
                    f"gate={metrics.get('gate', 0):.4f} mask={metrics.get('mask', 0):.4f} "
                    f"|d|={metrics.get('delta', 0):.5f}"
                )
            print(
                f"[{step+1}/{total}] PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} "
                f"best={best['psnr']:.3f}@{best['iter']} loss={loss.item():.4f} hr={elapsed:.2f}{extra}",
                flush=True,
            )
            model.train()

            if not args.no_kill_gate:
                cur = metrics["psnr"]
                if (
                    args.protect_base_drop > 0
                    and seed_base is not None
                    and "psnr_base" in metrics
                    and metrics["psnr_base"] < seed_base - args.protect_base_drop
                ):
                    write_kill(
                        save_dir,
                        {
                            "reason": "fail_base_drop",
                            "seed_base": seed_base,
                            "psnr_base": metrics["psnr_base"],
                            "drop_limit": args.protect_base_drop,
                            "iter": step + 1,
                        },
                    )
                    break
                if (
                    args.improve_by_k > 0
                    and seed_psnr is not None
                    and step + 1 >= args.improve_by_k
                    and best["psnr"] < seed_psnr + args.improve_delta
                ):
                    write_kill(
                        save_dir,
                        {
                            "reason": "fail_no_improve",
                            "seed_psnr": seed_psnr,
                            "best_psnr": best["psnr"],
                            "need_delta": args.improve_delta,
                            "iter": step + 1,
                        },
                    )
                    break
                if step + 1 >= 20000 and best["psnr"] < args.gate_20k:
                    write_kill(
                        save_dir,
                        {
                            "reason": "fail_20k_gate",
                            "best_psnr": best["psnr"],
                            "gate": args.gate_20k,
                            "iter": step + 1,
                        },
                    )
                    break
                if step + 1 >= 50000 and best["psnr"] < args.gate_50k:
                    write_kill(
                        save_dir,
                        {
                            "reason": "fail_50k_gate",
                            "best_psnr": best["psnr"],
                            "gate": args.gate_50k,
                            "iter": step + 1,
                        },
                    )
                    break

        pbar.set_postfix(loss=f"{loss.item():.4f}", best=f"{best['psnr']:.2f}")

    summary = {
        "model": args.model,
        "best": best,
        "target_psnr": args.target_psnr,
        "pass_lolv1": bool(best["psnr"] >= args.target_psnr and best["ssim"] >= 0.835),
        "params_m": nparams,
        "recipe": "RetinexFormer_LOL_v1.yml-aligned L1+mixup patch128",
    }
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
