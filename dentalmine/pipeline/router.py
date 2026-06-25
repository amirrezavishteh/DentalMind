"""ModalityRouter — decides which dental modality an input is.

Primary path: DICOM modality tag (0008,0060) + aspect-ratio heuristic.
Secondary path: Med-CLIP zero-shot classification on the image.
FMS: a folder of 14-18 PA-like images is treated as a full-mouth series.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from data.loaders.image_loader import LoadedImages
from models.medclip.base import MODALITY_PROMPTS
from schema.finding import ModalityType

_PROMPT_TO_MODALITY = {
    "bitewing": ModalityType.BW,
    "panoramic": ModalityType.OPG,
    "periapical": ModalityType.PA,
    "cbct": ModalityType.CBCT,
    "full_mouth_series": ModalityType.FMS,
}


class ModalityRouter:
    def __init__(self, medclip=None):
        # medclip is an optional MedCLIPEncoder (lazy — only used for fallback).
        self.medclip = medclip

    # ---- public --------------------------------------------------------
    def classify(
        self,
        loaded: LoadedImages,
        forced: Optional[ModalityType] = None,
    ) -> Tuple[ModalityType, float]:
        if forced is not None:
            return forced, 1.0

        # FMS: folder of many PA-sized images.
        if loaded.is_folder and 14 <= len(loaded.images) <= 18:
            return ModalityType.FMS, 0.7

        meta = loaded.dicom_meta or {}
        img = loaded.images[0]

        # Primary: DICOM tag.
        modality_tag = (meta.get("modality") or "").upper()
        if modality_tag in {"CT"} or meta.get("is_3d"):
            return ModalityType.CBCT, 0.9

        # For a single 2D image: aspect ratio is a strong, reliable prior.
        # Panoramic radiographs are unambiguously wide; only BW vs PA is
        # ambiguous, so Med-CLIP is used solely to disambiguate that band.
        h, w = img.shape[:2]
        ratio = w / max(h, 1)
        if ratio >= 1.7:
            return ModalityType.OPG, 0.9
        if self.medclip is not None:
            try:
                m, c = self._by_medclip(img, allowed=(ModalityType.BW, ModalityType.PA))
                if c >= 0.5:
                    return m, c
            except Exception:
                pass
        return self._by_aspect_ratio(img, base_conf=0.6)

    # ---- heuristics ----------------------------------------------------
    @staticmethod
    def _by_aspect_ratio(img: np.ndarray, base_conf: float) -> Tuple[ModalityType, float]:
        h, w = img.shape[:2]
        ratio = w / max(h, 1)
        if ratio > 2.0:                 # wide field of view
            return ModalityType.OPG, base_conf
        if ratio < 1.2:                 # square/tall, small FoV
            return ModalityType.PA, base_conf * 0.9
        return ModalityType.BW, base_conf * 0.8

    def _by_medclip(self, img: np.ndarray, allowed=None) -> Tuple[ModalityType, float]:
        items = list(MODALITY_PROMPTS.items())
        if allowed is not None:
            allowed_keys = {m for m in allowed}
            items = [
                (k, p) for k, p in items
                if _PROMPT_TO_MODALITY[k] in allowed_keys
            ]
        keys = [k for k, _ in items]
        prompts = [p for _, p in items]
        scores = self.medclip.zero_shot(img, prompts)
        best_i = int(np.argmax([scores[p] for p in prompts]))
        conf = float(scores[prompts[best_i]])
        return _PROMPT_TO_MODALITY[keys[best_i]], conf
