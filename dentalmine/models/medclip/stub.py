"""Deterministic offline stub Med-CLIP encoder.

Last-resort fallback when neither MMKD-CLIP weights nor BiomedCLIP (open_clip /
network) are available. Lets the full pipeline run end-to-end offline (e.g. in
CPU-only CI) by returning deterministic, image-content-derived pseudo-scores.

It is NOT a real model — scores are a hash-stable function of image statistics,
used only so downstream stages (C3/C4) have something to cluster. Always logged
loudly so it is never mistaken for a real encoder.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from models.medclip.base import DENTAL_PROMPTS, softmax


class StubEncoder:
    name = "Stub(offline)"
    image_embed_dim = 512
    text_embed_dim = 512

    def __init__(self, device: str = "cpu"):
        self.device = device

    def preprocess_image(self, pil_or_array):
        arr = np.asarray(pil_or_array, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return torch.from_numpy(arr).permute(2, 0, 1) / 255.0

    def _feat_from_image(self, image) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        # Derive a small deterministic feature from intensity stats.
        flat = arr.ravel()
        if flat.size == 0:
            return np.zeros(8, dtype=np.float32)
        stats = np.array([
            flat.mean(), flat.std(), np.median(flat),
            flat.min(), flat.max(),
            np.percentile(flat, 25), np.percentile(flat, 75),
            float(flat.size % 997),
        ], dtype=np.float32)
        return stats

    def encode_image(self, images) -> torch.Tensor:
        if isinstance(images, (list, tuple)):
            feats = np.stack([self._feat_from_image(im) for im in images])
        else:
            feats = self._feat_from_image(images)[None, :]
        t = torch.from_numpy(feats)
        return torch.nn.functional.normalize(t, dim=-1)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        rng = np.random.default_rng(0)
        feats = np.stack([
            rng.standard_normal(8).astype(np.float32) * 0 + (hash(t) % 100) / 100.0
            for t in texts
        ])
        return torch.from_numpy(feats)

    def zero_shot(self, image, text_prompts: List[str]) -> Dict[str, float]:
        # Deterministic pseudo-logits from image stats; stable per image.
        feat = self._feat_from_image(image)
        seed = int(abs(feat.sum())) % (2**32)
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal(len(text_prompts)).astype(np.float32)
        probs = softmax(logits)
        return {p: float(probs[i]) for i, p in enumerate(text_prompts)}

    def dental_zero_shot(self, image) -> Dict[str, float]:
        keys = list(DENTAL_PROMPTS.keys())
        scores = self.zero_shot(image, keys)
        return scores

    @classmethod
    def available(cls, *_a, **_k) -> bool:
        return True
