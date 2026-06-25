"""Med-CLIP factory — returns the best available encoder.

Priority:
  1. MMKD-CLIP   (if weights/MMKD_B16.pth exists)
  2. BiomedCLIP  (auto-downloads from HuggingFace; active path in Phase 1)
  3. Stub        (offline last resort so the pipeline still runs)
"""
from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

from paths import resolve

_console = Console()
_CACHE: dict = {}


class MedCLIPFactory:
    @staticmethod
    def get(config=None, device: str = "cuda", *, cache: bool = True):
        mmkd_path = None
        if config is not None:
            mc = config.get("medclip", {}) if hasattr(config, "get") else {}
            mmkd_path = mc.get("mmkd_weights", None)

        cache_key = (str(mmkd_path), device)
        if cache and cache_key in _CACHE:
            return _CACHE[cache_key]

        encoder = MedCLIPFactory._build(mmkd_path, device)
        if cache:
            _CACHE[cache_key] = encoder
        return encoder

    @staticmethod
    def _build(mmkd_path, device: str):
        # 0. Forced stub (CI / offline) via env var.
        if os.environ.get("DENTALMIND_MEDCLIP", "").lower() == "stub":
            from models.medclip.stub import StubEncoder
            _console.print("[yellow]Med-CLIP: Stub (forced via DENTALMIND_MEDCLIP=stub)[/yellow]")
            return StubEncoder(device="cpu")

        # 1. MMKD-CLIP
        if mmkd_path and Path(resolve(mmkd_path)).exists():
            try:
                from models.medclip.mmkd_clip import MMKDCLIPEncoder
                enc = MMKDCLIPEncoder(str(resolve(mmkd_path)), device)
                _console.print(f"[green]Med-CLIP: MMKD-CLIP[/green] ({mmkd_path})")
                return enc
            except Exception as e:  # pragma: no cover
                _console.print(f"[yellow]MMKD-CLIP load failed ({e}); trying BiomedCLIP[/yellow]")
        else:
            _console.print("[yellow]MMKD_B16.pth not found — falling back to BiomedCLIP[/yellow]")

        # 2. BiomedCLIP
        try:
            from models.medclip.biomedclip import BiomedCLIPEncoder
            enc = BiomedCLIPEncoder(device=device, freeze=True)
            _console.print("[green]Med-CLIP: BiomedCLIP[/green] (HuggingFace)")
            return enc
        except Exception as e:
            _console.print(
                f"[red]BiomedCLIP unavailable ({e}); using offline Stub encoder. "
                f"Detections will be pseudo-random — install open_clip_torch + network "
                f"for real results.[/red]"
            )

        # 3. Stub
        from models.medclip.stub import StubEncoder
        return StubEncoder(device="cpu")

    # convenience alias used in some PROMPT.md snippets
    @staticmethod
    def get_encoder(mmkd_path: str | None = None, device: str = "cuda"):
        return MedCLIPFactory._build(mmkd_path, device)


def get_best_medclip(config=None, device: str = "cuda"):
    return MedCLIPFactory.get(config, device)


def get_grounding_medclip(config=None, device: str = "cuda"):
    """Best Med-CLIP for C4 grounding. RegionMed-CLIP when released; else primary."""
    if config is not None:
        mc = config.get("medclip", {}) if hasattr(config, "get") else {}
        rm = mc.get("regionmed_weights", None)
        if rm and Path(resolve(rm)).exists():  # pragma: no cover - not released
            from models.medclip.regionmed_clip import RegionMedCLIPEncoder
            return RegionMedCLIPEncoder(str(resolve(rm)), device)
    return MedCLIPFactory.get(config, device)
