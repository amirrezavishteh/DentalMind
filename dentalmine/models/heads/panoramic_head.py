"""Panoramic (OPG) head — TWO-STAGE.

Stage 1 — tooth detection + FDI enumeration:
    * If a DENTEX-trained YOLO checkpoint is configured, run it and map class
      indices to FDI strings.
    * Otherwise (Phase 1 default) synthesize an anatomically-plausible 32-tooth
      FDI grid so the downstream integration (C3 clustering + C4 prompts) is
      fully exercised. TODO: replace with a DENTEX-trained YOLOv11 checkpoint.

Stage 2 — per-tooth pathology via Med-CLIP zero-shot:
    For each tooth ROI, run dental zero-shot and keep pathologies above
    threshold (excluding "healthy"). Emits Detection objects with tooth_fdi set.
    TODO: swap for fine-tuned HierarchicalDet when cloned into external/.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from models.medclip.base import DENTAL_PROMPTS
from schema.finding import Detection

# FDI numbering laid out left->right across a conventionally displayed OPG.
_UPPER_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
_LOWER_FDI = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

# Pathology prompts used in Stage 2 (exclude "healthy" from emitted findings).
_PATHOLOGY_KEYS = [k for k in DENTAL_PROMPTS if k != "healthy"]


class PanoramicHead:
    def __init__(
        self,
        medclip,
        stage1_weights: Optional[str] = None,
        stage1_model: str = "yolo11m.pt",
        score_threshold: float = 0.35,
        device: str = "cpu",
    ):
        self.medclip = medclip
        self.score_threshold = score_threshold
        self.device = device
        self.stage1_weights = stage1_weights
        self._yolo = None
        self._yolo_is_dental = False
        self._maybe_load_yolo(stage1_weights, stage1_model)

    def _maybe_load_yolo(self, weights: Optional[str], model_name: str):
        """Load ultralytics YOLO if a checkpoint is available.

        Only a dental-trained checkpoint (explicit ``weights``) is used for real
        FDI detection; the generic COCO model is not anatomically meaningful, so
        the synthetic FDI grid is used instead in that case.
        """
        if not weights:
            return  # use synthetic grid (placeholder)
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(weights)
            self._yolo_is_dental = True
        except Exception:
            self._yolo = None

    # ---- public --------------------------------------------------------
    def detect(self, display_bgr: np.ndarray, features=None) -> List[Detection]:
        """Run both stages and return per-tooth pathology detections."""
        tooth_boxes = self._stage1(display_bgr)
        return self._stage2(display_bgr, tooth_boxes)

    # ---- Stage 1 -------------------------------------------------------
    def _stage1(self, img: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
        """Return {fdi_str: bbox_xyxy normalized 0-1}."""
        if self._yolo is not None and self._yolo_is_dental:
            return self._stage1_yolo(img)
        return self._stage1_grid(img)

    def _stage1_yolo(self, img: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
        h, w = img.shape[:2]
        res = self._yolo.predict(img, verbose=False, device=self.device)[0]
        out: Dict[str, Tuple[float, float, float, float]] = {}
        names = res.names
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            fdi = str(names.get(cls_idx, cls_idx))
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out[fdi] = (x1 / w, y1 / h, x2 / w, y2 / h)
        return out

    @staticmethod
    def _stage1_grid(img: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
        """Synthetic 32-tooth FDI grid (placeholder).

        16 columns x 2 rows (upper/lower) of boxes spanning the arch.
        """
        out: Dict[str, Tuple[float, float, float, float]] = {}
        n = 16
        col_w = 1.0 / n
        box_w = col_w * 0.8
        pad = (col_w - box_w) / 2.0
        rows = {  # (y1, y2) normalized bands
            "upper": (0.10, 0.44),
            "lower": (0.56, 0.90),
        }
        for row_name, fdis in (("upper", _UPPER_FDI), ("lower", _LOWER_FDI)):
            y1, y2 = rows[row_name]
            for i, fdi in enumerate(fdis):
                x1 = i * col_w + pad
                x2 = x1 + box_w
                out[str(fdi)] = (x1, y1, x2, y2)
        return out

    # ---- Stage 2 -------------------------------------------------------
    def _stage2(
        self,
        img: np.ndarray,
        tooth_boxes: Dict[str, Tuple[float, float, float, float]],
    ) -> List[Detection]:
        h, w = img.shape[:2]
        detections: List[Detection] = []
        for fdi, (nx1, ny1, nx2, ny2) in tooth_boxes.items():
            x1, y1 = int(nx1 * w), int(ny1 * h)
            x2, y2 = int(nx2 * w), int(ny2 * h)
            roi = img[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                continue
            # BGR->RGB for the encoder.
            roi_rgb = roi[:, :, ::-1]
            try:
                scores = self.medclip.dental_zero_shot(roi_rgb)
            except Exception:
                continue
            for key in _PATHOLOGY_KEYS:
                s = float(scores.get(key, 0.0))
                if s >= self.score_threshold:
                    detections.append(Detection(
                        class_name=key,
                        confidence=s,
                        calibrated_confidence=s,   # calibrated later in postproc
                        bbox_xyxy=[nx1, ny1, nx2, ny2],
                        tooth_fdi=fdi,
                    ))
        return detections
