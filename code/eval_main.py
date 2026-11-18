"""Evaluate the main FSNA+GE checkpoint reported in the manuscript (LOL-v1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.retinexformer_wrap import FSNARFNet  # noqa: E402
from models.route_b_gen import FSNAGENNet  # noqa: E402
from utils.dataset import PairedLLDataset, resolve_split  # noqa: E402
from utils.metrics import psnr, ssim  # noqa: E402


def load_state(model, path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps = ss = n = 0.0
    for batch in tqdm(loader, desc="eval_main", ncols=100):
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        oh, ow = batch["orig_hw"][0].tolist()
        out = model(low)["enhanced"][:, :, :oh, :ow].clamp(0, 1)
        high = high[:, :, :oh, :ow]
        ps += psnr(out, high)
        ss += ssim(out, high)
        n += 1
    return {"psnr": ps / max(n, 1), "ssim": ss / max(n, 1), "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(ROOT.parent / "datasets"))
    ap.add_argument("--dataset", type=str, default="lolv1")
    ap.add_argument("--ckpt", type=str, default=str(ROOT.parent / "checkpoints" / "fsna_ge_lolv1_best.pth"))
    ap.add_argument("--fsna_init", type=str, default=str(ROOT.parent / "checkpoints" / "fsna_rf_lolv1_best.pth"))
    ap.add_argument("--detail_amp", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fsna = FSNARFNet().to(device)
    load_state(fsna, Path(args.fsna_init))
    for p in fsna.parameters():
        p.requires_grad = False
    fsna.eval()

    model = FSNAGENNet(fsna, detail_amp=args.detail_amp).to(device)
    missing, unexpected = load_state(model, Path(args.ckpt))

    pairs = resolve_split(args.data_root, args.dataset, "test")
    loader = DataLoader(PairedLLDataset(pairs, 128, train=False, photometric=False), batch_size=1)
    metrics = evaluate(model, loader, device)
    report = {
        "model": "fsna_ge",
        "dataset": args.dataset,
        "ckpt": str(args.ckpt),
        "fsna_init": str(args.fsna_init),
        "detail_amp": args.detail_amp,
        "missing": len(missing),
        "unexpected": len(unexpected),
        **metrics,
    }
    out = Path(args.out) if args.out else ROOT.parent / "results" / f"eval_main_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
