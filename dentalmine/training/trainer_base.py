"""Shared training utilities for the DentalMind phase scripts."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

import torch


def resolve_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def save_checkpoint(path: str | Path, **payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def list_2d_images(data_root: Path, max_images: int | None = None) -> List[Path]:
    """Collect all 2D radiograph image files under the standard dataset folders."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    search_dirs = [
        data_root / "tufts_yolo" / "images",
        data_root / "tufts" / "radiographs",
        data_root / "caries_zenodo" / "Dataset" / "x-ray" / "images",
        data_root / "mandible_mendeley" / "Images",
        data_root / "dentex",
    ]
    files: List[Path] = []
    for d in search_dirs:
        if d.exists():
            files.extend(p for p in d.rglob("*") if p.suffix.lower() in exts)
    files = sorted(set(files))
    if max_images:
        files = files[:max_images]
    return files


def maybe_limit_for_dev(items: Iterable, fast_dev_run: bool):
    """Return only the first batch's worth of items when --fast-dev-run is set."""
    items = list(items)
    return items[:1] if fast_dev_run else items
