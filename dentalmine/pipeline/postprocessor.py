"""PostProcessor — NMS, score filter, FDI + surface assignment, calibration.

Operates on Detection objects with normalized bboxes. For OPG the head already
sets tooth_fdi; surface is assigned from the detection's position within its
tooth box (here the detection == tooth box, so most resolve to 'body').
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from schema.finding import Detection, ModalityType


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PostProcessor:
    def __init__(
        self,
        nms_iou_threshold: float = 0.45,
        score_threshold: float = 0.35,
        calibration_temperature: float = 1.3,
        review_flag_threshold: float = 0.55,
    ):
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
        self.calibration_temperature = calibration_temperature
        self.review_flag_threshold = review_flag_threshold

    @classmethod
    def from_config(cls, cfg) -> "PostProcessor":
        inf = cfg.get("inference", {}) if hasattr(cfg, "get") else {}
        return cls(
            nms_iou_threshold=float(inf.get("nms_iou_threshold", 0.45)),
            score_threshold=float(inf.get("detection_score_threshold", 0.35)),
            calibration_temperature=float(inf.get("calibration_temperature", 1.3)),
            review_flag_threshold=float(inf.get("review_flag_threshold", 0.55)),
        )

    def process(
        self,
        raw_detections: List[Detection],
        modality: ModalityType,
        fdi_map: Optional[dict] = None,
        image_height_px: Optional[int] = None,
        audit_log: Optional[list] = None,
    ) -> List[Detection]:
        dets = self.nms(raw_detections)
        dets = self.score_filter(dets)
        dets = self.assign_fdi(dets, fdi_map, modality)
        dets = self.assign_surface(dets, fdi_map)
        dets = self.calibrate_confidence(dets)
        flagged = 0
        for d in dets:
            d.flagged_for_review = d.calibrated_confidence < self.review_flag_threshold
            flagged += int(d.flagged_for_review)
        if audit_log is not None:
            audit_log.append({"step": "postprocess", "flagged_count": flagged, "kept": len(dets)})
        return dets

    # ---- steps ---------------------------------------------------------
    def nms(self, dets: List[Detection]) -> List[Detection]:
        by_class: Dict[str, List[Detection]] = defaultdict(list)
        for d in dets:
            by_class[d.class_name].append(d)
        kept: List[Detection] = []
        for cls_dets in by_class.values():
            cls_dets = sorted(cls_dets, key=lambda d: -d.confidence)
            survivors: List[Detection] = []
            for d in cls_dets:
                if all(iou_xyxy(d.bbox_xyxy, s.bbox_xyxy) < self.nms_iou_threshold for s in survivors):
                    survivors.append(d)
            kept.extend(survivors)
        return kept

    def score_filter(self, dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.confidence >= self.score_threshold]

    @staticmethod
    def assign_fdi(dets: List[Detection], fdi_map, modality) -> List[Detection]:
        for i, d in enumerate(dets):
            if d.tooth_fdi:
                continue
            if modality == ModalityType.OPG and fdi_map:
                d.tooth_fdi = PostProcessor._nearest_fdi(d, fdi_map)
            else:
                d.tooth_fdi = f"unknown_{i+1}"
        return dets

    @staticmethod
    def _nearest_fdi(d: Detection, fdi_map: dict) -> str:
        cx, cy = d.bbox_center
        best, best_dist = None, 1e9
        for fdi, box in fdi_map.items():
            bx = (box[0] + box[2]) / 2.0
            by = (box[1] + box[3]) / 2.0
            dist = (cx - bx) ** 2 + (cy - by) ** 2
            if dist < best_dist:
                best, best_dist = fdi, dist
        return best or "unknown"

    @staticmethod
    def assign_surface(dets: List[Detection], fdi_map: Optional[dict]) -> List[Detection]:
        for d in dets:
            if d.surface:
                continue
            box = None
            if fdi_map and d.tooth_fdi in fdi_map:
                box = fdi_map[d.tooth_fdi]
            if box is None:
                continue
            tx1, ty1, tx2, ty2 = box
            tw = max(tx2 - tx1, 1e-6)
            th = max(ty2 - ty1, 1e-6)
            cx, cy = d.bbox_center
            x_rel = (cx - tx1) / tw
            y_rel = (cy - ty1) / th
            if x_rel < 0.3:
                d.surface = "mesial"
            elif x_rel > 0.7:
                d.surface = "distal"
            elif y_rel < 0.25:
                d.surface = "occlusal"
            elif y_rel > 0.75:
                d.surface = "cervical"
            else:
                d.surface = "body"
        return dets

    def calibrate_confidence(self, dets: List[Detection]) -> List[Detection]:
        t = self.calibration_temperature
        for d in dets:
            # temperature scaling on the logit of the confidence
            p = min(max(d.confidence, 1e-6), 1 - 1e-6)
            logit = np.log(p / (1 - p))
            d.calibrated_confidence = float(1.0 / (1.0 + np.exp(-logit / t)))
        return dets
