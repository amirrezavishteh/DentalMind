"""DentVFM-2D backbone — shared ViT encoder for all 2D modalities.

Weight priority (per PROMPT.md):
  1. DentVFM pretrained weights (if config path provided)
  2. timm ViT-L/14 DINOv2 (vit_large_patch14_dinov2)
  3. timm ViT (ImageNet) — last resort

For self-supervised pretraining from scratch (Phase 1) pass pretrained=False so
no download is needed. ``--model`` lets you drop to a tiny ViT for CPU smoke runs.

Outputs:
  encode_cls(x)      -> [B, D]              pooled CLS embedding
  encode_spatial(x)  -> [B, H', W', D]      patch grid (for detection heads / C1)
  encode(x)          -> [B, N, D]           flattened patch tokens
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

DEFAULT_MODEL = "vit_large_patch14_dinov2"


class DentVFMEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        weights_path: Optional[str] = None,
        pretrained: bool = True,
        freeze: bool = True,
        device: str = "cpu",
        img_size: Optional[int] = None,
    ):
        super().__init__()
        import timm

        kwargs = dict(pretrained=pretrained, num_classes=0)
        if img_size is not None:
            kwargs["img_size"] = img_size
        try:
            self.model = timm.create_model(model_name, **kwargs)
        except Exception:
            # network/weights unavailable -> random init (fine for SSL from scratch)
            kwargs["pretrained"] = False
            self.model = timm.create_model(model_name, **kwargs)

        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("model_state_dict", state)
            self.model.load_state_dict(state, strict=False)

        self.model_name = model_name
        self.embed_dim = getattr(self.model, "num_features", 768)
        self.patch_size = self._infer_patch_size()
        self.device = device
        self.to(device)
        if freeze:
            self.requires_grad_(False)
            self.model.eval()

    def _infer_patch_size(self) -> int:
        pe = getattr(self.model, "patch_embed", None)
        ps = getattr(pe, "patch_size", (14, 14)) if pe is not None else (14, 14)
        return ps[0] if isinstance(ps, (tuple, list)) else int(ps)

    # ---- feature extraction -------------------------------------------
    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens [B, N, D] (CLS/register tokens stripped)."""
        feats = self.model.forward_features(x)
        if feats.ndim == 3:
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            return feats[:, n_prefix:, :]
        if feats.ndim == 4:  # [B, D, H, W]
            b, d, h, w = feats.shape
            return feats.flatten(2).transpose(1, 2)
        raise RuntimeError(f"Unexpected feature shape {feats.shape}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._patch_tokens(x)

    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._patch_tokens(x)
        b, n, d = tokens.shape
        side = int(round(n ** 0.5))
        return tokens.reshape(b, side, side, d)

    def encode_cls(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        if feats.ndim == 3:
            return feats[:, 0, :]                 # CLS token
        return feats.mean(dim=(2, 3))             # GAP for conv-style features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_cls(x)
