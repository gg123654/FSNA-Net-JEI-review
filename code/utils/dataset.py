"""Paired low-light datasets with official splits."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    # SID/SDSD processed packs nest scene folders; recurse when top-level has no images.
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: str(p).lower())


def read_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        img = np.asarray(arr)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / (65535.0 if img.max() > 255.5 else 255.0)
        return np.clip(img[..., :3], 0.0, 1.0)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def paired_paths(low_dir: Path, high_dir: Path) -> list[tuple[Path, Path]]:
    lows = list_images(low_dir)
    highs = {p.name.lower(): p for p in list_images(high_dir)}
    # also allow stem match
    highs_stem = {p.stem.lower(): p for p in list_images(high_dir)}
    pairs = []
    for lp in lows:
        hp = highs.get(lp.name.lower())
        if hp is None:
            hp = highs_stem.get(lp.stem.lower())
        # LOL-v2-syn test: Low=rXXXX.png, Normal=XXXX.png (drop leading 'r')
        if hp is None:
            stem = lp.stem.lower()
            if stem.startswith("r") and stem[1:] in highs_stem:
                hp = highs_stem[stem[1:]]
        # also try lowXXXX <-> normalXXXX / highXXXX
        if hp is None:
            stem = lp.stem.lower()
            for pref in ("low", "lq", "input"):
                if stem.startswith(pref):
                    rest = stem[len(pref) :]
                    for alt in (f"normal{rest}", f"high{rest}", f"gt{rest}", rest):
                        if alt in highs_stem:
                            hp = highs_stem[alt]
                            break
                if hp is not None:
                    break
        if hp is None:
            continue
        pairs.append((lp, hp))
    if not pairs:
        raise RuntimeError(f"No pairs found under {low_dir} / {high_dir}")
    return pairs


class PairTransform:
    def __init__(self, patch_size: int = 256, train: bool = True, photometric: bool = True):
        self.patch_size = patch_size
        self.train = train
        self.photometric = photometric

    def __call__(self, low: np.ndarray, high: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = low.shape[:2]
        if self.train:
            ps = self.patch_size
            if h < ps or w < ps:
                scale = max(ps / h, ps / w)
                nh, nw = int(round(h * scale + 0.5)), int(round(w * scale + 0.5))
                low = cv2.resize(low, (nw, nh), interpolation=cv2.INTER_CUBIC)
                high = cv2.resize(high, (nw, nh), interpolation=cv2.INTER_CUBIC)
                h, w = low.shape[:2]
            y = np.random.randint(0, h - ps + 1)
            x = np.random.randint(0, w - ps + 1)
            low = low[y:y + ps, x:x + ps]
            high = high[y:y + ps, x:x + ps]
            if np.random.rand() < 0.5:
                low = np.flip(low, 1).copy()
                high = np.flip(high, 1).copy()
            if np.random.rand() < 0.5:
                low = np.flip(low, 0).copy()
                high = np.flip(high, 0).copy()
            k = np.random.randint(0, 4)
            if k:
                low = np.rot90(low, k).copy()
                high = np.rot90(high, k).copy()
            # Official RetinexFormer: geometric_augs only. Photometric joint scale
            # is legacy FSNA-v9 and hurts Route A recipe fidelity.
            if self.photometric and np.random.rand() < 0.4:
                scale = float(np.random.uniform(0.9, 1.1))
                low = np.clip(low * scale, 0.0, 1.0)
                high = np.clip(high * scale, 0.0, 1.0)
        else:
            # pad to multiple of 8 for network
            nh = (h + 7) // 8 * 8
            nw = (w + 7) // 8 * 8
            if nh != h or nw != w:
                low_p = np.zeros((nh, nw, 3), dtype=np.float32)
                high_p = np.zeros((nh, nw, 3), dtype=np.float32)
                low_p[:h, :w] = low
                high_p[:h, :w] = high
                low, high = low_p, high_p
        low_t = torch.from_numpy(low.transpose(2, 0, 1).copy())
        high_t = torch.from_numpy(high.transpose(2, 0, 1).copy())
        return low_t, high_t


class PairedLLDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        patch_size: int = 256,
        train: bool = True,
        photometric: bool = True,
    ):
        self.pairs = pairs
        self.tf = PairTransform(patch_size, train, photometric=photometric)
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        low = read_rgb(lp)
        high = read_rgb(hp)
        oh, ow = low.shape[:2]
        low_t, high_t = self.tf(low, high)
        return {
            "low": low_t,
            "high": high_t,
            "name": lp.name,
            "orig_hw": torch.tensor([oh, ow], dtype=torch.int32),
        }


class FullImagePairDataset(Dataset):
    """Full-resolution pairs for training the global gain head (v9).

    Long side is resized to <= max_side, then padded to a multiple of 8.
    Geometry augs only (flip/rot); no low-only photometric distortion.
    """

    def __init__(self, pairs: list[tuple[Path, Path]], max_side: int = 512, train: bool = True):
        self.pairs = pairs
        self.max_side = int(max_side)
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def _resize_long(self, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = low.shape[:2]
        m = max(h, w)
        if m > self.max_side:
            scale = self.max_side / float(m)
            nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            low = cv2.resize(low, (nw, nh), interpolation=cv2.INTER_AREA)
            high = cv2.resize(high, (nw, nh), interpolation=cv2.INTER_AREA)
        return low, high

    @staticmethod
    def _pad8(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        h, w = low.shape[:2]
        nh, nw = (h + 7) // 8 * 8, (w + 7) // 8 * 8
        if nh != h or nw != w:
            low_p = np.zeros((nh, nw, 3), dtype=np.float32)
            high_p = np.zeros((nh, nw, 3), dtype=np.float32)
            low_p[:h, :w] = low
            high_p[:h, :w] = high
            return low_p, high_p, h, w
        return low, high, h, w

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        low = read_rgb(lp)
        high = read_rgb(hp)
        oh, ow = low.shape[:2]
        low, high = self._resize_long(low, high)
        if self.train:
            if np.random.rand() < 0.5:
                low = np.flip(low, 1).copy()
                high = np.flip(high, 1).copy()
            if np.random.rand() < 0.5:
                low = np.flip(low, 0).copy()
                high = np.flip(high, 0).copy()
            k = np.random.randint(0, 4)
            if k:
                low = np.rot90(low, k).copy()
                high = np.rot90(high, k).copy()
            if np.random.rand() < 0.3:
                scale = float(np.random.uniform(0.9, 1.1))
                low = np.clip(low * scale, 0.0, 1.0)
                high = np.clip(high * scale, 0.0, 1.0)
        low, high, rh, rw = self._pad8(low, high)
        low_t = torch.from_numpy(low.transpose(2, 0, 1).copy())
        high_t = torch.from_numpy(high.transpose(2, 0, 1).copy())
        return {
            "low": low_t,
            "high": high_t,
            "name": lp.name,
            "orig_hw": torch.tensor([oh, ow], dtype=torch.int32),
            "resize_hw": torch.tensor([rh, rw], dtype=torch.int32),
        }


def resolve_split(root: str | Path, dataset: str, split: str) -> list[tuple[Path, Path]]:
    """Resolve official LOL-v1 / LOL-v2 paths."""
    root = Path(root)
    ds = dataset.lower()
    split = split.lower()
    candidates = []
    if ds in {"lol", "lolv1", "lol-v1"}:
        if split == "train":
            candidates = [
                (root / "LOLdataset" / "our485" / "low", root / "LOLdataset" / "our485" / "high"),
                (root / "LOLv1" / "our485" / "low", root / "LOLv1" / "our485" / "high"),
                (root / "LOL-v1" / "our485" / "low", root / "LOL-v1" / "our485" / "high"),
                (root / "LOLv1_en" / "train" / "low", root / "LOLv1_en" / "train" / "high"),
            ]
        else:
            candidates = [
                (root / "LOLdataset" / "eval15" / "low", root / "LOLdataset" / "eval15" / "high"),
                (root / "LOLv1" / "eval15" / "low", root / "LOLv1" / "eval15" / "high"),
                (root / "LOL-v1" / "eval15" / "low", root / "LOL-v1" / "eval15" / "high"),
                (root / "LOLv1_en" / "test" / "low", root / "LOLv1_en" / "test" / "high"),
            ]
    elif ds in {"lolv2-real", "lol-v2-real", "lolv2_real"}:
        if split == "train":
            candidates = [
                (root / "LOLv2" / "Real_captured" / "Train" / "Low", root / "LOLv2" / "Real_captured" / "Train" / "Normal"),
                (root / "LOL-v2" / "Real_captured" / "Train" / "Low", root / "LOL-v2" / "Real_captured" / "Train" / "Normal"),
                (root / "LOLv2" / "Real_captured" / "train" / "Low", root / "LOLv2" / "Real_captured" / "train" / "Normal"),
            ]
        else:
            candidates = [
                (root / "LOLv2" / "Real_captured" / "Test" / "Low", root / "LOLv2" / "Real_captured" / "Test" / "Normal"),
                (root / "LOL-v2" / "Real_captured" / "Test" / "Low", root / "LOL-v2" / "Real_captured" / "Test" / "Normal"),
                (root / "LOLv2" / "Real_captured" / "test" / "Low", root / "LOLv2" / "Real_captured" / "test" / "Normal"),
            ]
    elif ds in {"lolv2-syn", "lol-v2-syn", "lolv2_synthetic", "lol-v2-synthetic"}:
        if split == "train":
            candidates = [
                (root / "LOLv2" / "Synthetic" / "Train" / "Low", root / "LOLv2" / "Synthetic" / "Train" / "Normal"),
                (root / "LOL-v2" / "Synthetic" / "Train" / "Low", root / "LOL-v2" / "Synthetic" / "Train" / "Normal"),
            ]
        else:
            candidates = [
                (root / "LOLv2" / "Synthetic" / "Test" / "Low", root / "LOLv2" / "Synthetic" / "Test" / "Normal"),
                (root / "LOL-v2" / "Synthetic" / "Test" / "Low", root / "LOL-v2" / "Synthetic" / "Test" / "Normal"),
            ]
    else:
        raise ValueError(dataset)

    for low_d, high_d in candidates:
        if low_d.is_dir() and high_d.is_dir():
            return paired_paths(low_d, high_d)
    raise FileNotFoundError(f"Cannot resolve {dataset}/{split} under {root}")


class UnpairedFolder(Dataset):
    def __init__(self, folder: str | Path):
        self.files = list_images(folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = read_rgb(p)
        h, w = img.shape[:2]
        nh, nw = (h + 7) // 8 * 8, (w + 7) // 8 * 8
        if nh != h or nw != w:
            pad = np.zeros((nh, nw, 3), dtype=np.float32)
            pad[:h, :w] = img
            img = pad
        t = torch.from_numpy(img.transpose(2, 0, 1).copy())
        return {"low": t, "name": p.name, "orig_hw": torch.tensor([h, w], dtype=torch.int32)}
