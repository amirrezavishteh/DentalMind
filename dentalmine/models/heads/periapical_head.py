"""Periapical (PA) head.

PROMPT.md target: YOLOv8s + Swin-T hybrid (periapical lesion, caries, resorption,
root-filling adequacy, etc.). Until a trained checkpoint + PA dataset exist, this
is a Med-CLIP zero-shot region detector restricted to PA-relevant pathologies
(apical lesions, deep caries, bone loss). A PA shows 1-3 teeth with their roots
and apices, so the grid is near-square.

TODO: replace with a trained YOLOv8s+Swin-T head.
"""
from __future__ import annotations

from models.heads.zeroshot_region_head import ZeroShotRegionHead

# PAs capture the apex/periapex — periapical lesions are the key target.
_PA_KEYS = ["periapical_lesion", "deep_caries", "dentin_caries", "bone_loss"]


class PeriapicalHead(ZeroShotRegionHead):
    def __init__(self, medclip, score_threshold: float = 0.35, device: str = "cpu"):
        super().__init__(medclip, _PA_KEYS, score_threshold,
                         grid_x=2, grid_y=2, device=device)
