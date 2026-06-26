"""MMKD-CLIP encoder — PRIMARY Med-CLIP for the whole pipeline.

MMKD-CLIP (ViT-B-16-quickgelu) distills 9 biomedical CLIPs; #1 AUC on X-ray.
  Source : github.com/wangshansong1/MMKD-CLIP  (clone to external/MMKD-CLIP)
  Weights: gdown 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR -O weights/MMKD_B16.pth

Activates automatically once ``weights/MMKD_B16.pth`` exists and the source
repo is checked out at ``external/MMKD-CLIP``. Until then MedCLIPFactory uses
BiomedCLIP.

NOTE: this loader follows the official README exactly but has NOT been run
locally in Phase 1 (weights not yet downloaded). API matches BiomedCLIPEncoder.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.medclip.base import DENTAL_PROMPTS, softmax
from paths import PROJECT_ROOT

DEFAULT_WEIGHTS = "./weights/MMKD_B16.pth"
# The MMKD repo vendors its own open_clip fork as a top-level package
# (external/MMKD-CLIP/open_clip/), with extra exports (get_mean_std, HFTokenizer)
# not present in the public pip open_clip_torch. Prepend the repo root itself
# (not src/) so `import open_clip` resolves to the fork, and drop any
# already-imported public open_clip module so it doesn't shadow it.
_MMKD_REPO_ROOT = PROJECT_ROOT / "external" / "MMKD-CLIP"


def _ensure_mmkd_open_clip_on_path():
    repo_root = str(_MMKD_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    else:
        sys.path.remove(repo_root)
        sys.path.insert(0, repo_root)
    # If the public open_clip_torch package was already imported (e.g. by
    # BiomedCLIPEncoder running first), purge it so the fork is loaded fresh.
    for mod_name in list(sys.modules):
        if mod_name == "open_clip" or mod_name.startswith("open_clip."):
            del sys.modules[mod_name]
    importlib.invalidate_caches()


class MMKDCLIPEncoder:
    name = "MMKD-CLIP"
    TEXT_ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    MODEL_NAME = "ViT-B-16-quickgelu"
    CONTEXT_LEN = 256
    image_embed_dim = 512
    text_embed_dim = 512

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, device: str = "cuda"):
        if not _MMKD_REPO_ROOT.exists():
            raise ImportError(
                f"MMKD-CLIP repo not found at {_MMKD_REPO_ROOT}. "
                f"git clone https://github.com/wangshansong1/MMKD-CLIP {_MMKD_REPO_ROOT}"
            )
        _ensure_mmkd_open_clip_on_path()
        from open_clip import create_model_and_transforms, get_mean_std, HFTokenizer

        self.device = device
        mean, std = get_mean_std()
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=self.MODEL_NAME,
            pretrained=weights_path,
            precision="amp",
            device=device,
            force_quick_gelu=True,
            mean=mean,
            std=std,
            inmem=True,
            text_encoder_name=self.TEXT_ENCODER,
        )
        self.tokenizer = HFTokenizer(self.TEXT_ENCODER, context_length=self.CONTEXT_LEN)
        self.model.eval()

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
        with torch.autocast("cuda", enabled=(self.device == "cuda")):
            feats = self.model.encode_image(batch)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = torch.cat([self.tokenizer(t) for t in texts], dim=0).to(self.device)
        with torch.autocast("cuda", enabled=(self.device == "cuda")):
            feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    # ---- zero-shot -----------------------------------------------------
    @torch.no_grad()
    def zero_shot(self, image, text_prompts: List[str]) -> Dict[str, float]:
        img_f = self.encode_image(image)
        txt_f = self.encode_text(text_prompts)
        logit_scale = self.model.logit_scale.exp()
        logits = (logit_scale * img_f @ txt_f.t())[0]
        probs = softmax(logits.float().cpu().numpy())
        return {p: float(probs[i]) for i, p in enumerate(text_prompts)}

    @torch.no_grad()
    def dental_zero_shot(self, image) -> Dict[str, float]:
        keys = list(DENTAL_PROMPTS.keys())
        prompts = list(DENTAL_PROMPTS.values())
        scores = self.zero_shot(image, prompts)
        return {keys[i]: scores[prompts[i]] for i in range(len(keys))}

    @classmethod
    def available(cls, weights_path: str = DEFAULT_WEIGHTS) -> bool:
        return bool(weights_path) and os.path.exists(weights_path)
