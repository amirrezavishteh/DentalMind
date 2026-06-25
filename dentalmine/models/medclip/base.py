"""Shared Med-CLIP interface + dental prompt vocabulary.

All encoders (MMKD-CLIP primary, BiomedCLIP fallback, Stub offline-safe)
implement this interface so the factory can swap them transparently.
"""
from __future__ import annotations

from typing import Dict, List, Protocol

import numpy as np

# Canonical dental pathology prompt set used for grounding / zero-shot.
DENTAL_PROMPTS: Dict[str, str] = {
    "enamel_caries": "dental caries in enamel on bitewing radiograph",
    "dentin_caries": "dentin caries near pulp on dental X-ray",
    "deep_caries": "deep caries near pulp on dental radiograph",
    "periapical_lesion": "periapical lesion on endodontic radiograph",
    "bone_loss": "alveolar bone loss on dental radiograph",
    "impaction": "impacted tooth on panoramic X-ray",
    "calculus": "calculus deposit on dental radiograph",
    "healthy": "healthy tooth on dental radiograph",
}

# Modality classification prompts (router secondary path).
MODALITY_PROMPTS: Dict[str, str] = {
    "bitewing": "bitewing dental X-ray showing interproximal teeth",
    "panoramic": "panoramic dental X-ray showing full mouth",
    "periapical": "periapical X-ray showing tooth root and apex",
    "cbct": "dental CBCT cone beam CT cross-section",
    "full_mouth_series": "full mouth series dental radiograph",
}


class MedCLIPEncoder(Protocol):
    """Structural interface every Med-CLIP encoder implements."""

    name: str

    def encode_image(self, images): ...  # -> Tensor[B, D] (L2-normalized)
    def encode_text(self, texts: List[str]): ...  # -> Tensor[N, D]
    def preprocess_image(self, pil_or_array): ...  # -> Tensor[3, H, W]
    def zero_shot(self, image, text_prompts: List[str]) -> Dict[str, float]: ...
    def dental_zero_shot(self, image) -> Dict[str, float]: ...


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-9)
