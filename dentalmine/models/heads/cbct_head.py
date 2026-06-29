"""CBCT head — per-slice detection on the shared CLIP encoder.

Phase-1-style zero-shot detector: each axial slice is divided into a coarse grid
of ROIs; the shared Med-CLIP encoder scores each ROI with dental prompts and any
pathology above threshold becomes a Detection tagged with its slice_idx. Those
per-slice detections then flow into C1 (neighbour fusion) and C2 (neighbour-
consistency decision) in the inference engine.

TODO: replace zero-shot grid with a trained 2.5D head consuming C1-enriched
features (Phase 2) once CBCT training data exists.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.medclip.base import DENTAL_PROMPTS
from schema.finding import Detection

# CBCT-relevant pathologies (exclude tooth-position/enamel-only cues).
_CBCT_KEYS = [
    "periapical_lesion", "bone_loss", "deep_caries", "dentin_caries", "impaction",
]


class CBCTHead:
    def __init__(self, medclip, score_threshold: float = 0.35, grid: int = 3, device: str = "cpu"):
        self.medclip = medclip
        self.score_threshold = score_threshold
        self.grid = grid
        self.device = device

    def detect_slice(self, slice_bgr: np.ndarray, slice_idx: int) -> List[Detection]:
        h, w = slice_bgr.shape[:2]
        g = self.grid
        dets: List[Detection] = []
        for gy in range(g):
            for gx in range(g):
                x1, y1 = gx / g, gy / g
                x2, y2 = (gx + 1) / g, (gy + 1) / g
                roi = slice_bgr[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
                if roi.size == 0:
                    continue
                try:
                    scores = self.medclip.dental_zero_shot(roi[:, :, ::-1])  # BGR->RGB
                except Exception:
                    continue
                for key in _CBCT_KEYS:
                    s = float(scores.get(key, 0.0))
                    if s >= self.score_threshold:
                        dets.append(Detection(
                            class_name=key, confidence=s, calibrated_confidence=s,
                            bbox_xyxy=[x1, y1, x2, y2], slice_idx=slice_idx,
                        ))
        return dets

    def detect_volume(self, display_slices: List[np.ndarray]) -> Dict[int, List[Detection]]:
        """Run per-slice detection across the whole volume -> {slice_idx: [Detection]}."""
        return {i: self.detect_slice(sl, i) for i, sl in enumerate(display_slices)}
