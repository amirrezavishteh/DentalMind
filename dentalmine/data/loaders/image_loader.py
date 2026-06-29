"""Simple 2D image / folder loader for Phase 1 (OPG, BW, PA).

Returns numpy uint8 images (H,W,3) plus a metadata dict. DICOM is read via
pydicom when available; CBCT/3D volume loading lives in cbct_loader (later phase).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DICOM_EXTS = {".dcm", ".dicom"}


@dataclass
class LoadedImages:
    images: List[np.ndarray]                 # list of HxWx3 uint8
    paths: List[str]
    is_folder: bool = False
    dicom_meta: dict = field(default_factory=dict)
    slice_spacing: Optional[float] = None


def _read_one(path: Path) -> tuple[np.ndarray, dict]:
    ext = path.suffix.lower()
    if ext in DICOM_EXTS:
        return _read_dicom(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # cv2 fails on some 16-bit pngs; fall back to PIL.
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]  # to BGR
    return np.ascontiguousarray(img), {}


def _read_dicom(path: Path) -> tuple[np.ndarray, dict]:
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    meta = {
        "modality": getattr(ds, "Modality", None),
        "rows": int(getattr(ds, "Rows", arr.shape[0])),
        "cols": int(getattr(ds, "Columns", arr.shape[1])),
    }
    if "PixelSpacing" in ds:
        try:
            meta["pixel_spacing"] = float(ds.PixelSpacing[0])
        except Exception:
            pass
    return np.ascontiguousarray(arr), meta


class ImageLoader:
    """Loads a single image file or a folder of images."""

    def load(self, input_path: str) -> LoadedImages:
        p = Path(input_path)
        if p.is_dir():
            files = sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in IMAGE_EXTS | DICOM_EXTS
            )
            if not files:
                raise FileNotFoundError(f"No images found in folder: {p}")
            images, paths, metas = [], [], []
            for f in files:
                img, meta = _read_one(f)
                images.append(img)
                paths.append(str(f))
                metas.append(meta)
            return LoadedImages(
                images=images, paths=paths, is_folder=True,
                dicom_meta=metas[0] if metas else {},
            )
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
        img, meta = _read_one(p)
        return LoadedImages(images=[img], paths=[str(p)], dicom_meta=meta)
