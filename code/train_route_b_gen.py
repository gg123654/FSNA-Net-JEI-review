"""Route B probe: train RF-GEN (frozen RF + learned global exposure) and gate.

Discipline (measure-then-train): a <=30min probe with a hard pass gate. Long
training is only justified if `pass_probe=true` (best >= seed + pass_delta).

Target: beat the RF seed (25.16 on lolv1) by >= 0.25 dB to be worth a long run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.route_b_gen import RFGENNet, count_params  # noqa: E402
from utils.dataset import FullImagePairDataset, PairedLLDataset, resolve_split  # noqa: E402
from utils.losses import ssim_loss  # noqa: E402
from utils.metrics import psnr, ssim  # noqa: E402


def cycle(loader):
    while True:
        for b in loader:
            yield b


def pad_collate(batch):
    maxh = max(b["low"].shape[1] for b in batch)
    maxw = max(b["low"].shape[2] for b in batch)
    lows, highs = [], []
    for b in batch:
        l, h = b["low"], b["high"]
        ph, pw = maxh - l.shape[1], maxw - l.shape[2]
        lows.append(F.pad(l, (0, pw, 0, ph), mode="replicate"))
        highs.append(F.pad(h, (0, pw, 0, ph), mode="replicate"))
    return {"low": torch.stack(lows), "high": torch.stack(highs)}


def mixup(low, high, beta=1.2, p=0.5):
    if torch.rand(1).item() > p:
        return low, high
    lam = float(torch.distributions.Beta(beta, beta).sample())
    idx = torch.randperm(low.size(0), device=low.device)
    return lam * low + (1 - lam) * low[idx], lam * high + (1 - lam) * high[idx]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    p = s = 0.0
    n = 0
    for b in loader:
        low = b["low"].to(device)
        gt = b["high"].to(device)
        oh, ow = b["orig_hw"][0].tolist()
        out = model(low)["enhanced"][:, :, :oh, :ow].clamp(0, 1)
        gt = gt[:, :, :oh, :ow]
        p += psnr(out, gt)
        s += ssim(out, gt)
        n += 1
    return {"psnr": p / max(n, 1), "ssim": s / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT.parent / "datasets"))
    ap.add_argument("--dataset", default="lolv1")
    ap.add_argument("--rf_weights", default="")
    ap.add_argument("--iters", type=int, default=1800)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_side", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--mixup_p", type=float, default=0.0, help="0 = no mixup (global exposure dislikes exposure blending)")
    ap.add_argument("--detail_amp", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pass_delta", type=float, default=0.15)
    ap.add_argument("--base", choices=["rf", "fsna"], default="rf")
    ap.add_argument("--fsna_init", default="")
    ap.add_argument("--tag", default="route_b_probe_gene_lolv1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wpath = Path(args.rf_weights) if args.rf_weights else (
        ROOT.parent / "third_party" / "Retinexformer" / "pretrained_weights" / "LOL_v1.pth"
    )

    if args.base == "fsna":
        from models.retinexformer_wrap import FSNARFNet
        from models.route_b_gen import FSNAGENNet
        fpath = Path(args.fsna_init) if args.fsna_init else (
            ROOT.parent / "checkpoints" / "route_a_fsna_rf_lolv1" / "best.pth"
        )
        fsna = FSNARFNet().to(device)
        fck = torch.load(fpath, map_location="cpu", weights_only=False)
        fst = fck["model"] if isinstance(fck, dict) and "model" in fck else fck
        fm, fu = fsna.load_state_dict(fst, strict=False)
        for p in fsna.parameters():
            p.requires_grad = False
        fsna.eval()
        model = FSNAGENNet(fsna, detail_amp=args.detail_amp).to(device)
        print(f"FSNA base loaded {fpath.name} miss={len(fm)} unexp={len(fu)} "
              f"trainable(GE)={count_params(model):.3f}M", flush=True)
    else:
        model = RFGENNet(detail_amp=args.detail_amp).to(device)
        ck = torch.load(wpath, map_location="cpu", weights_only=False)
        st = ck["params"] if isinstance(ck, dict) and "params" in ck else ck
        if isinstance(st, dict) and "params_ema" in st:
            st = st["params_ema"]
        miss, unexp = model.backbone.load_state_dict(st, strict=False)
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"RF base loaded {wpath.name} miss={len(miss)} unexp={len(unexp)} "
              f"trainable(GE)={count_params(model):.3f}M", flush=True)

    train_pairs = resolve_split(args.data_root, args.dataset, "train")
    test_pairs = resolve_split(args.data_root, args.dataset, "test")
    tr_ds = FullImagePairDataset(train_pairs, max_side=args.max_side, train=True)
    te_ds = PairedLLDataset(test_pairs, 128, train=False, photometric=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
                       collate_fn=pad_collate)
    te_ld = DataLoader(te_ds, batch_size=1, shuffle=False, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    seed = evaluate(model, te_ld, device)
    print(f"SEED (identity == RF): PSNR={seed['psnr']:.3f} SSIM={seed['ssim']:.4f}", flush=True)
    best = {"psnr": seed["psnr"], "ssim": seed["ssim"], "iter": 0}
    hist = [{"iter": 0, **seed}]

    data = cycle(tr_ld)
    t0 = time.time()
    ck_dir = ROOT.parent / "checkpoints" / args.tag
    ck_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.iters + 1):
        model.train()
        if step <= args.warmup:
            cur_lr = args.lr * step / max(args.warmup, 1)
        else:
            t = (step - args.warmup) / max(args.iters - args.warmup, 1)
            cur_lr = 0.05 * args.lr + 0.95 * args.lr * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        b = next(data)
        low = b["low"].to(device, non_blocking=True)
        high = b["high"].to(device, non_blocking=True)
        if args.mixup_p > 0:
            low, high = mixup(low, high, p=args.mixup_p)
        out = model(low)["enhanced"]
        loss = F.l1_loss(out, high) + 0.2 * ssim_loss(out, high)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.iters:
            m = evaluate(model, te_ld, device)
            hist.append({"iter": step, **m, "loss": float(loss.item())})
            tag = ""
            if m["psnr"] > best["psnr"]:
                best = {"psnr": m["psnr"], "ssim": m["ssim"], "iter": step}
                torch.save({"model": model.state_dict(), "iter": step, "metrics": best}, ck_dir / "best.pth")
                tag = " *best*"
            el = (time.time() - t0) / 60.0
            print(f"[{step}/{args.iters}] PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
                  f"best={best['psnr']:.3f} loss={loss.item():.4f} {el:.1f}min{tag}", flush=True)
            if step == args.iters:
                with torch.no_grad():
                    gb = model(low)
                print(f"  mean gain~{['%.3f'%x for x in gb['gain'].mean(0).tolist()]} "
                      f"gamma~{['%.3f'%x for x in gb['gamma'].mean(0).tolist()]}", flush=True)

    delta = best["psnr"] - seed["psnr"]
    pass_probe = bool(best["psnr"] >= seed["psnr"] + args.pass_delta)
    out = {
        "route": "B_global_exposure",
        "dataset": args.dataset,
        "seed_psnr": round(seed["psnr"], 4),
        "seed_ssim": round(seed["ssim"], 4),
        "best": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
        "delta_vs_seed": round(delta, 4),
        "pass_delta": args.pass_delta,
        "pass_probe": pass_probe,
        "iters": args.iters,
        "minutes": round((time.time() - t0) / 60.0, 1),
        "history": hist,
        "verdict": (
            "PROMISING: long-train worth it" if pass_probe
            else "FAIL: global-exposure family does not beat RF enough; do NOT long-train"
        ),
    }
    out_path = Path(args.out) if args.out else (ROOT.parent / "results" / "route_b_probe_gene.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "iter": args.iters, "metrics": hist[-1]}, ck_dir / "last.pth")
    print("DONE", {k: out[k] for k in ("seed_psnr", "best", "delta_vs_seed", "pass_probe", "verdict")}, flush=True)


if __name__ == "__main__":
    main()
