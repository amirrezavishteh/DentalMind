"""BiomedCLIP encoder — FALLBACK Med-CLIP (active when MMKD weights absent).

Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
ViT-B/16 image encoder + PubMedBERT text encoder, trained on PMC-15M.
Auto-downloads from HuggingFace on first use (~430 MB).

API matches MMKDCLIPEncoder so MedCLIPFactory can swap transparently.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.medclip.base import DENTAL_PROMPTS, softmax

HF_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BiomedCLIPEncoder:
    name = "BiomedCLIP"
    image_embed_dim = 512
    text_embed_dim = 512

    def __init__(self, device: str = "cuda", freeze: bool = True, context_length: int = 256):
        from open_clip import create_model_from_pretrained, get_tokenizer

        self.device = device
        self.context_length = context_length
        self.model, self.preprocess = create_model_from_pretrained(f"hf-hub:{HF_ID}")
        self.tokenizer = get_tokenizer(f"hf-hub:{HF_ID}")
        self.model.to(device).eval()
        if freeze:
            self.model.requires_grad_(False)

    # ---- preprocessing -------------------------------------------------
    def preprocess_image(self, pil_or_array) -> torch.Tensor:
        if isinstance(pil_or_array, np.ndarray):
            arr = pil_or_array
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            pil = Image.fromarray(arr.astype(np.uint8))
        elif isinstance(pil_or_array, Image.Image):
            pil = pil_or_array.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type {type(pil_or_array)}")
        return self.preprocess(pil)

    def _as_batch(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            t = images if images.ndim == 4 else images.unsqueeze(0)
        elif isinstance(images, (list, tuple)):
            t = torch.stack([self.preprocess_image(im) for im in images])
        else:
            t = self.preprocess_image(images).unsqueeze(0)
        return t.to(self.device)

    # ---- encoders ------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        batch = self._as_batch(images)
        feats = self.model.encode_image(batch)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts, context_length=self.context_length).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    # ---- zero-shot -----------------------------------------------------
    @torch.no_grad()
    def zero_shot(self, image, text_prompts: List[str]) -> Dict[str, float]:
        img_f = self.encode_image(image)               # [1, D]
        txt_f = self.encode_text(text_prompts)         # [N, D]
        logit_scale = self.model.logit_scale.exp()
        logits = (logit_scale * img_f @ txt_f.t())[0]  # [N]
        probs = softmax(logits.float().cpu().numpy())
        return {p: float(probs[i]) for i, p in enumerate(text_prompts)}

    @torch.no_grad()
    def dental_zero_shot(self, image) -> Dict[str, float]:
        keys = list(DENTAL_PROMPTS.keys())
        prompts = list(DENTAL_PROMPTS.values())
        scores = self.zero_shot(image, prompts)
        # remap from prompt text back to canonical keys
        return {keys[i]: scores[prompts[i]] for i in range(len(keys))}

    @classmethod
    def available(cls, *_args, **_kwargs) -> bool:
        try:
            import open_clip  # noqa: F401
            return True
        except ImportError:
            return False
