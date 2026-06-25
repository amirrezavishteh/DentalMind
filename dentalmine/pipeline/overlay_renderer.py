"""OverlayRenderer — color-coded 2D overlays + export of result artifacts.

Phase 1: 2D rendering + export of findings.json / original.png / annotated.png /
treatment_card.txt / summary.txt. CBCT 3D mesh rendering is deferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from schema.finding import InferenceResult, ToothCluster
from schema.prompt import PromptOutput

# BGR colors per finding class.
COLOR_MAP = {
    "enamel_caries":        (80, 80, 255),
    "dentin_caries":        (40, 40, 220),
    "deep_caries":          (40, 40, 180),
    "secondary_caries":     (120, 120, 255),
    "bone_loss":            (30, 160, 255),
    "furcation_defect":     (50, 200, 255),
    "periapical_lesion":    (50, 230, 255),
    "impaction":            (240, 90, 160),
    "calculus":             (255, 160, 100),
    "existing_restoration": (120, 120, 120),
}
_DEFAULT_COLOR = (200, 200, 200)

SEVERITY_COLORS = {
    "CRITICAL": (0, 0, 255),
    "HIGH": (0, 128, 255),
    "MEDIUM": (0, 255, 255),
    "LOW": (0, 255, 0),
}


class OverlayRenderer:
    def __init__(self, overlay_alpha: float = 0.30, show_fdi_labels: bool = True):
        self.overlay_alpha = overlay_alpha
        self.show_fdi_labels = show_fdi_labels

    # ---- rendering -----------------------------------------------------
    def render_2d(self, display_bgr: np.ndarray, clusters: List[ToothCluster]) -> np.ndarray:
        out = display_bgr.copy()
        h, w = out.shape[:2]
        overlay = out.copy()
        for cluster in clusters:
            sev_color = SEVERITY_COLORS.get(cluster.severity.value, _DEFAULT_COLOR)
            first_box = None
            for f in cluster.findings:
                color = COLOR_MAP.get(f.class_name, _DEFAULT_COLOR)
                x1, y1, x2, y2 = self._denorm(f.bbox_xyxy, w, h)
                if first_box is None:
                    first_box = (x1, y1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # filled for alpha
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if first_box is not None:
                lx, ly = first_box
                label = f"{cluster.tooth_fdi} | {cluster.severity.value}"
                if self.show_fdi_labels:
                    cv2.putText(out, label, (lx, max(12, ly - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, sev_color, 1, cv2.LINE_AA)
                cv2.circle(out, (lx - 8, max(8, ly - 8)), 5, sev_color, -1)
                if any(f.flagged_for_review for f in cluster.findings):
                    self._dashed_rect(out, self._denorm(cluster.findings[0].bbox_xyxy, w, h), sev_color)
                    cv2.putText(out, "? REVIEW", (lx, min(h - 4, ly + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, self.overlay_alpha, out, 1 - self.overlay_alpha, 0, out)
        return out

    def render(self, images, clusters):
        """Render the first/primary image (Phase 1 = single 2D image)."""
        img = images[0] if isinstance(images, list) else images
        return self.render_2d(img, clusters)

    # ---- export --------------------------------------------------------
    def export(
        self,
        result: InferenceResult,
        clusters: List[ToothCluster],
        prompts: List[PromptOutput],
        original_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        output_dir: str,
    ) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out / "original.png"), original_bgr)
        cv2.imwrite(str(out / "annotated.png"), annotated_bgr)

        with open(out / "findings.json", "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))

        card = "\n\n".join(p.to_text() for p in prompts) if prompts else (
            "No findings.\n\n" + (prompts[0].disclaimer if prompts else
            "⚠ AI second opinion only. "
            "Final diagnosis and treatment plan subject to clinical judgment.")
        )
        with open(out / "treatment_card.txt", "w", encoding="utf-8") as fh:
            fh.write(card)

        with open(out / "summary.txt", "w", encoding="utf-8") as fh:
            fh.write(result.to_report())

        return {
            "findings_json": str(out / "findings.json"),
            "original_png": str(out / "original.png"),
            "annotated_png": str(out / "annotated.png"),
            "treatment_card_txt": str(out / "treatment_card.txt"),
            "summary_txt": str(out / "summary.txt"),
        }

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _denorm(bbox, w, h):
        x1, y1, x2, y2 = bbox
        return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

    @staticmethod
    def _dashed_rect(img, box, color, dash=6):
        x1, y1, x2, y2 = box
        for x in range(x1, x2, dash * 2):
            cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, 1)
            cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, 1)
        for y in range(y1, y2, dash * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, 1)
            cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, 1)
