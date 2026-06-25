"""MadClip C3 — cluster per-detection outputs into compound per-tooth findings.

Strategy 1 (default): group detections by FDI, derive a clinical pattern from
the set of class_names on each tooth, assign severity/urgency from PATTERN_LIBRARY.
Strategy 2 (fallback when FDI confidence is low): DBSCAN on spatial+class features.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from schema.finding import Detection, Severity, ToothCluster, Urgency

PATTERN_LIBRARY: Dict[str, Dict[str, str]] = {
    "caries_enamel":     {"severity": "LOW",      "urgency": "MONITOR"},
    "caries_dentin":     {"severity": "HIGH",     "urgency": "SOON"},
    "caries_pulp_risk":  {"severity": "CRITICAL", "urgency": "IMMEDIATE"},
    "caries_with_bone":  {"severity": "HIGH",     "urgency": "SOON"},
    "caries_with_pal":   {"severity": "CRITICAL", "urgency": "IMMEDIATE"},
    "bone_loss_only":    {"severity": "MEDIUM",   "urgency": "ROUTINE"},
    "multi_tooth_bone":  {"severity": "HIGH",     "urgency": "SOON"},
    "impaction":         {"severity": "MEDIUM",   "urgency": "ROUTINE"},
    "endo_complex":      {"severity": "CRITICAL", "urgency": "IMMEDIATE"},
    "calculus_only":     {"severity": "LOW",      "urgency": "ROUTINE"},
    "restoration_issue": {"severity": "MEDIUM",   "urgency": "ROUTINE"},
}

_URGENCY_MAP = {
    "IMMEDIATE": Urgency.IMMEDIATE,
    "SOON": Urgency.SOON,
    "ROUTINE": Urgency.ROUTINE,
    "MONITOR": Urgency.MONITOR,
}


class FindingClusterer:
    def __init__(self, fdi_confidence_threshold: float = 0.65, px_spacing_mm: float = 0.1):
        self.fdi_confidence_threshold = fdi_confidence_threshold
        self.px_spacing_mm = px_spacing_mm

    # ---- public --------------------------------------------------------
    def cluster(
        self,
        detections: List[Detection],
        image_height_px: Optional[int] = None,
        fdi_confidence: float = 1.0,
    ) -> List[ToothCluster]:
        if not detections:
            return []
        if fdi_confidence < self.fdi_confidence_threshold or any(
            d.tooth_fdi is None for d in detections
        ):
            clusters = self.cluster_by_dbscan(detections, image_height_px)
        else:
            clusters = self.cluster_by_fdi(detections, image_height_px)
        return self._apply_multi_tooth_bone(clusters)

    # ---- Strategy 1 ----------------------------------------------------
    def cluster_by_fdi(
        self, detections: List[Detection], image_height_px: Optional[int] = None
    ) -> List[ToothCluster]:
        groups: Dict[str, List[Detection]] = defaultdict(list)
        for d in detections:
            groups[d.tooth_fdi or "unknown"].append(d)
        return [self._build_cluster(fdi, dets, image_height_px) for fdi, dets in groups.items()]

    # ---- Strategy 2 ----------------------------------------------------
    def cluster_by_dbscan(
        self, detections: List[Detection], image_height_px: Optional[int] = None
    ) -> List[ToothCluster]:
        from sklearn.cluster import DBSCAN

        feats = []
        class_vocab = sorted({d.class_name for d in detections})
        for d in detections:
            cx, cy = d.bbox_center
            emb = np.zeros(len(class_vocab), dtype=np.float32)
            emb[class_vocab.index(d.class_name)] = 1.0
            feats.append(np.concatenate([[cx, cy], emb]))
        X = np.array(feats)
        labels = DBSCAN(eps=0.12, min_samples=1).fit_predict(X)

        groups: Dict[int, List[Detection]] = defaultdict(list)
        for lbl, d in zip(labels, detections):
            groups[int(lbl)].append(d)
        clusters = []
        for i, (lbl, dets) in enumerate(groups.items()):
            fdi = dets[0].tooth_fdi or f"unknown_{i+1}"
            clusters.append(self._build_cluster(fdi, dets, image_height_px))
        return clusters

    # ---- cluster builder ----------------------------------------------
    def _build_cluster(
        self, fdi: str, dets: List[Detection], image_height_px: Optional[int]
    ) -> ToothCluster:
        class_names = {d.class_name for d in dets}
        pattern = self._pattern_from_classes(class_names)
        spec = PATTERN_LIBRARY[pattern]

        surfaces = sorted({d.surface for d in dets if d.surface})
        bone_loss_mm = self._estimate_bone_loss(dets, image_height_px)
        conf = float(np.mean([d.calibrated_confidence for d in dets]))

        return ToothCluster(
            tooth_fdi=fdi,
            severity=Severity(spec["severity"]),
            urgency=_URGENCY_MAP[spec["urgency"]],
            pattern=pattern,
            findings=dets,
            cluster_confidence=conf,
            surfaces_affected=surfaces,
            bone_loss_mm=bone_loss_mm,
        )

    @staticmethod
    def _pattern_from_classes(classes: set) -> str:
        has_pal = "periapical_lesion" in classes
        has_caries = bool(classes & {"enamel_caries", "dentin_caries", "deep_caries", "secondary_caries"})
        has_deep = "deep_caries" in classes
        has_bone = "bone_loss" in classes
        has_impaction = "impaction" in classes
        has_calculus = "calculus" in classes
        has_restoration = "overhanging_filling" in classes or "restoration_issue" in classes

        if has_pal and has_caries:
            return "caries_with_pal"
        if has_pal:
            return "endo_complex"
        if has_deep:
            return "caries_pulp_risk"
        if has_caries and has_bone:
            return "caries_with_bone"
        if "dentin_caries" in classes or "secondary_caries" in classes:
            return "caries_dentin"
        if "enamel_caries" in classes:
            return "caries_enamel"
        if has_impaction:
            return "impaction"
        if has_bone:
            return "bone_loss_only"
        if has_restoration:
            return "restoration_issue"
        if has_calculus:
            return "calculus_only"
        # default: treat as enamel caries (lowest-severity caries) if nothing matched
        return "caries_enamel"

    def _estimate_bone_loss(
        self, dets: List[Detection], image_height_px: Optional[int]
    ) -> Optional[float]:
        bone = [d for d in dets if d.class_name == "bone_loss"]
        if not bone or image_height_px is None:
            return None
        # bbox height (normalized) * image height (px) * px_spacing (mm/px)
        heights_mm = [
            (d.bbox_xyxy[3] - d.bbox_xyxy[1]) * image_height_px * self.px_spacing_mm
            for d in bone
        ]
        return round(float(max(heights_mm)), 1)

    # ---- cross-tooth pass ---------------------------------------------
    @staticmethod
    def _apply_multi_tooth_bone(clusters: List[ToothCluster]) -> List[ToothCluster]:
        bone_clusters = [c for c in clusters if c.pattern == "bone_loss_only"]
        if len(bone_clusters) >= 3:
            spec = PATTERN_LIBRARY["multi_tooth_bone"]
            for c in bone_clusters:
                c.pattern = "multi_tooth_bone"
                c.severity = Severity(spec["severity"])
                c.urgency = _URGENCY_MAP[spec["urgency"]]
        return clusters

    def merge_with_3d(self, clusters_2d, nnunet_output):  # pragma: no cover - later phase
        raise NotImplementedError("3D merge requires the CBCT/nnU-Net phase.")
