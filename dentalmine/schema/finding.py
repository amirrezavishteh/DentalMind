"""Core finding / cluster / result schemas (Pydantic v2).

These are the contract every pipeline stage produces and consumes; implement
first — everything depends on them (PROMPT.md: "SCHEMAS — implement first").
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Urgency(str, Enum):
    IMMEDIATE = "immediate_24h"
    SOON = "soon_2_weeks"
    ROUTINE = "routine_next_visit"
    MONITOR = "monitor_6_months"


class ModalityType(str, Enum):
    BW = "bitewing"
    SERIAL_BW = "serial_bitewing"
    OPG = "panoramic"
    PA = "periapical"
    FMS = "full_mouth_series"
    CBCT = "cbct"


# Ordering used to compute the "highest" severity across a result.
_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def severity_rank(sev: Severity) -> int:
    return _SEVERITY_ORDER[Severity(sev)]


# CLI-friendly modality aliases -> ModalityType.
_MODALITY_ALIASES = {
    "bw": ModalityType.BW,
    "opg": ModalityType.OPG,
    "pa": ModalityType.PA,
    "fms": ModalityType.FMS,
    "cbct": ModalityType.CBCT,
    "serial_bw": ModalityType.SERIAL_BW,
}


def parse_modality(s) -> ModalityType:
    """Accept either an alias ('opg') or an enum value ('panoramic')."""
    if isinstance(s, ModalityType):
        return s
    key = str(s).lower().strip()
    if key in _MODALITY_ALIASES:
        return _MODALITY_ALIASES[key]
    return ModalityType(key)


class Detection(BaseModel):
    class_name: str
    confidence: float
    calibrated_confidence: float
    bbox_xyxy: List[float]            # [x1,y1,x2,y2] normalized 0-1
    mask_rle: Optional[str] = None    # RLE encoded mask
    surface: Optional[str] = None     # mesial/distal/occlusal/cervical/buccal
    slice_idx: Optional[int] = None   # CBCT slice index
    flagged_for_review: bool = False
    tooth_fdi: Optional[str] = None   # set by postprocessor.assign_fdi

    @property
    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class ToothCluster(BaseModel):
    tooth_fdi: str                    # "26", "unknown_1"
    severity: Severity
    urgency: Urgency
    pattern: str                      # key from PATTERN_LIBRARY
    findings: List[Detection]
    cluster_confidence: float
    surfaces_affected: List[str]
    bone_loss_mm: Optional[float] = None
    has_3d_mask: bool = False
    neighbor_support_score: float = 0.0   # C2 output


class InferenceResult(BaseModel):
    modality: ModalityType
    image_paths: List[str]
    clusters: List[ToothCluster] = Field(default_factory=list)
    total_teeth_affected: int = 0
    highest_severity: Severity = Severity.LOW
    processing_time_ms: float = 0.0
    model_versions: dict = Field(default_factory=dict)
    patient_id: Optional[str] = None
    audit_log: List[dict] = Field(default_factory=list)
    error: Optional[str] = None       # set when QualityError / pipeline failure

    def recompute_summary(self) -> "InferenceResult":
        """Fill total_teeth_affected and highest_severity from clusters."""
        self.total_teeth_affected = len(self.clusters)
        if self.clusters:
            self.highest_severity = max(
                (c.severity for c in self.clusters), key=severity_rank
            )
        else:
            self.highest_severity = Severity.LOW
        return self

    def to_report(self) -> str:
        """Human-readable summary for treatment_card / summary.txt."""
        if self.error:
            return f"Inference failed: {self.error}"
        if not self.clusters:
            return (
                f"Modality: {self.modality.value}. No pathology detected "
                f"across the image(s). Routine review recommended."
            )
        lines = [
            f"Modality: {self.modality.value}",
            f"Teeth affected: {self.total_teeth_affected}",
            f"Highest severity: {self.highest_severity.value}",
            "",
        ]
        for c in sorted(self.clusters, key=lambda x: -severity_rank(x.severity)):
            surfaces = ", ".join(c.surfaces_affected) if c.surfaces_affected else "n/a"
            lines.append(
                f"  Tooth {c.tooth_fdi}: {c.pattern} "
                f"[{c.severity.value} / {c.urgency.value}] "
                f"surfaces={surfaces} conf={c.cluster_confidence:.2f}"
            )
        return "\n".join(lines)
