"""Evaluate FSNA-RF / RetinexFormer checkpoints on paired splits (measured only)."""

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

from models.retinexformer_wrap import (  # noqa: E402
    FSNARFNet,
    build_retinexformer,
    count_rf_parameters,
)
from utils.dataset import PairedLLDataset, resolve_split  # noqa: E402
from utils.metrics import psnr, ssim  # noqa: E402


def build_model(kind: str, device, freeze_bb: bool = False):
    if kind == "retinexformer":
        model = build_retinexformer(n_feat=40, stage=1, num_blocks=(1, 2, 2)).to(device)
        return model, False
    if kind == "fsna_rf":
        model = FSNARFNet(
            n_feat=40,
            stage=1,
            num_blocks=(1, 2, 2),
            use_frb=True,
            use_naf=True,
            freeze_backbone=freeze_bb,
        ).to(device)
        return model, True
    if kind == "fsna_rf_no_frb":
        model = FSNARFNet(
            n_feat=40,
            stage=1,
            num_blocks=(1, 2, 2),
            use_frb=False,
            use_naf=False,
            freeze_backbone=freeze_bb,
        ).to(device)
        return model, True
    if kind == "fsna_rf_no_naf":
        model = FSNARFNet(
            n_feat=40,
            stage=1,
            num_blocks=(1, 2, 2),
            use_frb=True,
            use_naf=False,
            freeze_backbone=freeze_bb,
        ).to(device)
        return model, True
    raise ValueError(kind)


def load_weights(model, path: Path, is_dict: bool, official_rf: bool = False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if official_rf:
        state = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
    else:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


@torch.no_grad()
def evaluate(model, loader, device, is_dict: bool):
    model.eval()
    ps = ss = n = 0.0
    extras = {"psnr_base": 0.0, "psnr_fused": 0.0, "gate": 0.0, "mask": 0.0}
    has_extra = False
    for batch in tqdm(loader, desc="eval", ncols=100):
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        oh, ow = batch["orig_hw"][0].tolist()
        out = model(low)
        pred = out["enhanced"] if is_dict else out
        pred = pred[:, :, :oh, :ow].clamp(0, 1)
        high = high[:, :, :oh, :ow]
        ps += psnr(pred, high)
        ss += ssim(pred, high)
        n += 1
        if is_dict and isinstance(out, dict) and "base" in out:
            has_extra = True
            extras["psnr_base"] += psnr(out["base"][:, :, :oh, :ow].clamp(0, 1), high)
            if "fused" in out:
                extras["psnr_fused"] += psnr(out["fused"][:, :, :oh, :ow].clamp(0, 1), high)
            if "gate" in out:
                extras["gate"] += float(out["gate"].mean().item())
            if "mask" in out:
                extras["mask"] += float(out["mask"].mean().item())
    out_m = {"psnr": ps / max(n, 1), "ssim": ss / max(n, 1), "n": int(n)}
    if has_extra:
        for k in extras:
            out_m[k] = extras[k] / max(n, 1)
    return out_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="fsna_rf")
    ap.add_argument("--dataset", type=str, default="lolv1")
    ap.add_argument("--data_root", type=str, default=str(ROOT.parent / "datasets"))
    ap.add_argument("--official_rf", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, is_dict = build_model(args.model, device)
    missing, unexpected = load_weights(model, Path(args.ckpt), is_dict, official_rf=args.official_rf)
    pairs = resolve_split(args.data_root, args.dataset, "test")
    loader = DataLoader(PairedLLDataset(pairs, 128, train=False, photometric=False), batch_size=1)
    metrics = evaluate(model, loader, device, is_dict)
    report = {
        "tag": args.tag or Path(args.ckpt).stem,
        "model": args.model,
        "dataset": args.dataset,
        "ckpt": str(args.ckpt),
        "params_m": count_rf_parameters(model),
        "missing": len(missing),
        "unexpected": len(unexpected),
        **metrics,
    }
    out = Path(args.out) if args.out else ROOT.parent / "results" / f"eval_{report['tag']}_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
