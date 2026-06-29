"""Bitewing (BW) head.

PROMPT.md target: YOLOv8x-seg with DentVFM feature injection (8 caries/perio
classes). Until a trained checkpoint + bitewing dataset exist, this is a Med-CLIP
zero-shot region detector restricted to bitewing-relevant pathologies
(interproximal caries, bone loss, calculus). A bitewing shows only a few
posterior crowns, so the grid is wide and shallow.

TODO: replace with a trained YOLOv8x-seg head consuming DentVFM features.
"""
from __future__ import annotations

from models.heads.zeroshot_region_head import ZeroShotRegionHead

# Bitewings image crowns/interproximal surfaces + crestal bone — not apices.
_BW_KEYS = ["enamel_caries", "dentin_caries", "deep_caries", "bone_loss", "calculus"]


class BitewingHead(ZeroShotRegionHead):
    def __init__(self, medclip, score_threshold: float = 0.35, device: str = "cpu"):
        super().__init__(medclip, _BW_KEYS, score_threshold,
                         grid_x=4, grid_y=2, device=device)
