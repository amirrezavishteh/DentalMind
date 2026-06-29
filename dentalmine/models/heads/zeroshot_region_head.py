"""Shared zero-shot region detector for 2D modalities without a trained head.

Divides the image into a grid of ROIs and scores each with the shared Med-CLIP
encoder against a modality-specific pathology prompt set; ROIs above threshold
become Detection objects. Used as the runnable placeholder for the bitewing and
periapical heads until trained detectors exist.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from schema.finding import Detection


class ZeroShotRegionHead:
    def __init__(self, medclip, pathology_keys: Sequence[str],
                 score_threshold: float = 0.35, grid_x: int = 4, grid_y: int = 2,
                 device: str = "cpu"):
        self.medclip = medclip
        self.pathology_keys = list(pathology_keys)
        self.score_threshold = score_threshold
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.device = device

    def detect(self, display_bgr: np.ndarray, features=None) -> List[Detection]:
        h, w = display_bgr.shape[:2]
        gx, gy = self.grid_x, self.grid_y
        dets: List[Detection] = []
        for iy in range(gy):
            for ix in range(gx):
                nx1, ny1 = ix / gx, iy / gy
                nx2, ny2 = (ix + 1) / gx, (iy + 1) / gy
                roi = display_bgr[int(ny1 * h):int(ny2 * h), int(nx1 * w):int(nx2 * w)]
                if roi.size == 0:
                    continue
                try:
                    scores = self.medclip.dental_zero_shot(roi[:, :, ::-1])  # BGR->RGB
                except Exception:
                    continue
                for key in self.pathology_keys:
                    s = float(scores.get(key, 0.0))
                    if s >= self.score_threshold:
                        dets.append(Detection(
                            class_name=key, confidence=s, calibrated_confidence=s,
                            bbox_xyxy=[nx1, ny1, nx2, ny2],
                        ))
        return dets
