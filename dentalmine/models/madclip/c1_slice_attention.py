"""MadClip C1 — cross-slice attention adapter.

Gives the frozen 2D DentVFM encoder 3D awareness for CBCT / FMS / serial-BW by
letting the central slice's patch features attend over the patch features of its
K neighbours. ~2M params; trained in Phase 2 with the backbone frozen.

forward(features_list, target_idx) -> [B, H, W, D]
  features_list : list of (2K+1) spatial feature maps, each [B, H, W, D]
  target_idx    : index of the central slice within the list
Single-slice input returns the central features unchanged (identity).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CrossSliceAttentionAdapter(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features_list: List[torch.Tensor], target_idx: int) -> torch.Tensor:
        if len(features_list) == 1:
            return features_list[0]                       # identity for single 2D

        central = features_list[target_idx]               # [B, H, W, D]
        b, h, w, d = central.shape
        n = h * w

        query = central.reshape(b, n, d)                  # [B, N, D]
        kv = torch.stack(features_list, dim=1)            # [B, S, H, W, D]
        kv = kv.reshape(b, len(features_list) * n, d)     # [B, S*N, D]

        attended, _ = self.attn(query, kv, kv)            # [B, N, D]
        out = self.norm(self.proj(attended))
        out = out + query                                 # residual on central
        return out.reshape(b, h, w, d)

    @staticmethod
    def select_k(modality: str, slice_spacing_mm: float = 1.0) -> int:
        """Neighbour half-window K by modality (see PROMPT.md)."""
        m = str(modality).lower()
        if "cbct" in m:
            return max(2, round(1.0 / max(slice_spacing_mm, 1e-3)))
        if "fms" in m or "full_mouth" in m:
            return 1
        if "serial" in m:
            return 2
        return 0                                           # single 2D -> identity
