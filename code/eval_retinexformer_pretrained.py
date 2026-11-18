"""Download (optional) and eval official RetinexFormer domain weights — no GT-mean.

Supports LOL-v1 / LOL-v2-real / LOL-v2-syn. Official CaiYuanhao weights only
(NOT M2Retinexformer mirrors — those are a different arch).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.retinexformer_wrap import build_retinexformer, count_rf_parameters
from utils.dataset import PairedLLDataset, resolve_split
from utils.metrics import psnr, ssim

WEIGHT_DIR = ROOT.parent / "third_party" / "Retinexformer" / "pretrained_weights"

# Official CaiYuanhao weights (NOT M2Retinexformer mirrors — those are a different arch).
DOMAIN = {
    "lolv1": {
        "file": "LOL_v1.pth",
        "urls": [
            "https://hf-mirror.com/CaiYuanhao/Retinexformer/resolve/main/LOL_v1.pth",
            "https://huggingface.co/CaiYuanhao/Retinexformer/resolve/main/LOL_v1.pth",
        ],
        "dataset": "lolv1",
        "out": "retinexformer_pretrained_lolv1.json",
        "anchor_public": {"psnr": 25.16, "ssim": 0.845},
        "gate": {"psnr": 24.8, "ssim": 0.835},
    },
    "lolv2-real": {
        "file": "LOL_v2_real.pth",
        "urls": [
            "https://hf-mirror.com/CaiYuanhao/Retinexformer/resolve/main/LOL_v2_real.pth",
            "https://huggingface.co/CaiYuanhao/Retinexformer/resolve/main/LOL_v2_real.pth",
        ],
        "dataset": "lolv2-real",
        "out": "retinexformer_pretrained_lolv2_real.json",
        "anchor_public": {"psnr": 22.80, "ssim": 0.840},
        "gate": {"psnr": 22.5, "ssim": 0.830},
    },
    "lolv2-syn": {
        "file": "LOL_v2_synthetic.pth",
        "urls": [
            "https://hf-mirror.com/CaiYuanhao/Retinexformer/resolve/main/LOL_v2_synthetic.pth",
            "https://huggingface.co/CaiYuanhao/Retinexformer/resolve/main/LOL_v2_synthetic.pth",
        ],
        "dataset": "lolv2-syn",
        "out": "retinexformer_pretrained_lolv2_syn.json",
        "anchor_public": {"psnr": 25.67, "ssim": 0.930},
        "gate": {"psnr": 25.2, "ssim": 0.920},
    },
}


def download(urls: list[str], dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Official RF ckpts are ~6–8MB; reject known wrong mirrors (~400MB M2).
    if dest.exists() and 5_000_000 < dest.stat().st_size < 20_000_000:
        print(f"exists: {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        return
    last_err = None
    for u in urls:
        try:
            print(f"Downloading {u} -> {dest}", flush=True)
            urllib.request.urlretrieve(u, dest)
            sz = dest.stat().st_size
            print(f"saved {sz/1e6:.1f} MB", flush=True)
            if not (5_000_000 < sz < 20_000_000):
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"Unexpected weight size {sz} — likely wrong mirror")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"download failed: {e}", flush=True)
            if dest.exists():
                dest.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download weights: {last_err}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT.parent / "datasets"))
    ap.add_argument(
        "--dataset",
        default="lolv1",
        choices=list(DOMAIN.keys()),
        help="Domain to evaluate (selects official weights + split)",
    )
    ap.add_argument("--weights", default="", help="Override weight path")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="", help="Override JSON out path")
    args = ap.parse_args()

    meta = DOMAIN[args.dataset]
    wpath = Path(args.weights) if args.weights else (WEIGHT_DIR / meta["file"])
    out_path = Path(args.out) if args.out else (ROOT.parent / "results" / meta["out"])

    if args.download or not wpath.exists():
        download(meta["urls"], wpath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_retinexformer(n_feat=40, stage=1, num_blocks=(1, 2, 2)).to(device)
    ckpt = torch.load(wpath, map_location=device, weights_only=False)
    state = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
    if isinstance(state, dict) and "params_ema" in state:
        state = state["params_ema"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"load missing={len(missing)} unexpected={len(unexpected)} "
        f"params={count_rf_parameters(model):.3f}M",
        flush=True,
    )

    pairs = resolve_split(args.data_root, meta["dataset"], "test")
    loader = DataLoader(PairedLLDataset(pairs, 128, train=False), batch_size=1, shuffle=False)
    model.eval()
    psnr_v = ssim_v = n = 0.0
    per = []
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        oh, ow = batch["orig_hw"][0].tolist()
        pred = model(low)[:, :, :oh, :ow].clamp(0, 1)
        high = high[:, :, :oh, :ow]
        pv, sv = psnr(pred, high), ssim(pred, high)
        psnr_v += pv
        ssim_v += sv
        n += 1
        per.append({"name": batch["name"][0], "psnr": pv, "ssim": sv})
    out = {
        "protocol": "standard_psnr_ssim_no_gt_mean",
        "dataset": meta["dataset"],
        "weights": str(wpath),
        "psnr": psnr_v / n,
        "ssim": ssim_v / n,
        "n": int(n),
        "anchor_public": meta["anchor_public"],
        "gate": meta["gate"],
        "per_image": per,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(
        json.dumps(
            {k: out[k] for k in ("dataset", "psnr", "ssim", "n", "protocol", "weights")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
