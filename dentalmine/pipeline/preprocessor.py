"""Preprocessor — 2D (CLAHE + crop + normalize) and CBCT slice preprocessing.

preprocess_2d  -> YOLO-sized + DentVFM-sized tensors for a single 2D image.
preprocess_cbct -> per-slice uint8 RGB images (for the shared CLIP encoder)
                   + physical slice spacing, for the 3D inference path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from schema.finding import ModalityType

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Preprocessed2D:
    # Image used for detection / display (uint8 BGR, cropped & CLAHE'd).
    display_bgr: np.ndarray
    # Normalized CHW float tensors at two resolutions.
    yolo_chw: np.ndarray        # [3, 640, 640]
    dentvfm_chw: np.ndarray     # [3, 518, 518]
    # Crop box applied to the original image: (x1, y1, x2, y2) in original px.
    crop_box: Tuple[int, int, int, int]
    orig_shape: Tuple[int, int]  # (H, W) of the original image


class Preprocessor:
    def __init__(self, yolo_size: int = 640, dentvfm_size: int = 518):
        self.yolo_size = yolo_size
        self.dentvfm_size = dentvfm_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_2d(
        self,
        image: np.ndarray,
        modality: ModalityType = ModalityType.OPG,
        snr_db: float | None = None,
    ) -> Preprocessed2D:
        orig_h, orig_w = image.shape[:2]

        # 1. grayscale -> 3-channel
        if image.ndim == 2:
            gray = image
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2. CLAHE on luminance
        gray_eq = self.clahe.apply(gray)
        bgr = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

        # 3. foreground crop via Otsu + 10px pad
        x1, y1, x2, y2 = self._foreground_box(gray_eq, pad=10)
        bgr_crop = bgr[y1:y2, x1:x2]
        if bgr_crop.size == 0:
            bgr_crop = bgr
            x1, y1, x2, y2 = 0, 0, orig_w, orig_h

        # 6. PA denoise if low SNR (ESRGAN placeholder; TODO: replace)
        if modality == ModalityType.PA and snr_db is not None and snr_db < 20:
            g = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
            g = cv2.fastNlMeansDenoising(g, h=7)
            bgr_crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        yolo_chw = self._normalize_resize(bgr_crop, self.yolo_size)
        dentvfm_chw = self._normalize_resize(bgr_crop, self.dentvfm_size)

        return Preprocessed2D(
            display_bgr=bgr_crop,
            yolo_chw=yolo_chw,
            dentvfm_chw=dentvfm_chw,
            crop_box=(x1, y1, x2, y2),
            orig_shape=(orig_h, orig_w),
        )

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _foreground_box(gray: np.ndarray, pad: int = 10) -> Tuple[int, int, int, int]:
        h, w = gray.shape
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, w, h
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(w, int(xs.max()) + pad)
        y2 = min(h, int(ys.max()) + pad)
        return x1, y1, x2, y2

    @staticmethod
    def _normalize_resize(bgr: np.ndarray, size: int) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    # ---- CBCT (3D) -----------------------------------------------------
    def preprocess_cbct(self, slices, spacing_mm: float = 1.0):
        """Per-slice normalization for the 3D path.

        Returns (display_slices, spacing_mm) where each display slice is a uint8
        RGB image suitable for the shared CLIP encoder / overlay rendering. The
        volume is z-score normalized as a whole (preserves cross-slice intensity
        relationships that C2 relies on), then per-slice min-max stretched to 8-bit.
        """
        vol = np.stack([np.asarray(s, dtype=np.float32) for s in slices], axis=0)
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)
        display = []
        for sl in vol:
            mn, mx = float(sl.min()), float(sl.max())
            g = (sl - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(sl)
            g8 = self.clahe.apply(g.astype(np.uint8))
            display.append(cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR))
        return display, spacing_mm
