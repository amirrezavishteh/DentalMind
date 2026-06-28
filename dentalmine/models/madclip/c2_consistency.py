"""MadClip C2 — neighbour-consistency false-positive filter.

Pure inference-time, no training. A detection on slice N is kept only if the
same class appears (IoU >= threshold) on enough physically-nearby slices —
real lesions persist across adjacent slices, noise does not. Single 2D images
are returned unchanged.
"""
from __future__ import annotations

from typing import Dict, List

from pipeline.postprocessor import iou_xyxy
from schema.finding import Detection


class ConsistencyFilter:
    def __init__(
        self,
        min_support: int = 3,
        iou_threshold: float = 0.40,
        physical_window_mm: float = 1.0,
        confidence_boost: float = 0.08,
    ):
        self.min_support = min_support
        self.iou_threshold = iou_threshold
        self.physical_window_mm = physical_window_mm
        self.confidence_boost = confidence_boost

    @classmethod
    def from_config(cls, cfg) -> "ConsistencyFilter":
        inf = cfg.get("inference", {}) if hasattr(cfg, "get") else {}
        return cls(
            min_support=int(inf.get("c2_min_support", 3)),
            iou_threshold=float(inf.get("c2_consistency_iou", 0.40)),
            physical_window_mm=float(inf.get("c2_physical_window_mm", 1.0)),
        )

    def filter(
        self,
        detections_per_image: Dict[int, List[Detection]],
        spacing_mm: float = 1.0,
    ) -> Dict[int, List[Detection]]:
        if len(detections_per_image) <= 1:
            return detections_per_image                    # single slice -> unchanged

        window_slices = max(1, round(self.physical_window_mm / max(spacing_mm, 1e-3)))
        out: Dict[int, List[Detection]] = {}
        for idx, dets in detections_per_image.items():
            kept = []
            neighbour_idxs = [
                j for j in detections_per_image
                if j != idx and abs(j - idx) <= window_slices
            ]
            for d in dets:
                support = 0
                for j in neighbour_idxs:
                    if any(
                        nb.class_name == d.class_name
                        and iou_xyxy(d.bbox_xyxy, nb.bbox_xyxy) >= self.iou_threshold
                        for nb in detections_per_image[j]
                    ):
                        support += 1
                if support >= self.min_support:
                    boost = 1.0 + self.confidence_boost * support
                    d.calibrated_confidence = min(1.0, d.calibrated_confidence * boost)
                    kept.append(d)
            out[idx] = kept
        return out
