"""Batch evaluation — run the engine over a folder and aggregate a report.

This is an HONEST inference report (per-image cluster/severity/timing stats), NOT
a detection-accuracy (mAP) benchmark: that requires ground-truth boxes, which are
not wired into a labelled eval split here. For Stage-1 tooth-detector mAP, use
ultralytics directly (`yolo detect val model=best.pt data=tufts_yolo/data.yaml`).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from config_loader import load_config

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def evaluate_dir(test_dir: str, modality: str = "auto", config_path: Optional[str] = None,
                 checkpoint: Optional[str] = None, device: Optional[str] = None) -> dict:
    from pipeline.inference_engine import InferenceEngine

    cfg = load_config(config_path)
    engine = InferenceEngine(cfg, checkpoint_dir=checkpoint, device=device)

    files = sorted(p for p in Path(test_dir).rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found under {test_dir}")

    severity_hist = Counter()
    n_errors = 0
    total_clusters = 0
    total_ms = 0.0
    per_image = []

    for fp in files:
        r = engine.infer(str(fp), modality=modality)
        if r.error:
            n_errors += 1
            per_image.append({"image": fp.name, "error": r.error})
            continue
        total_clusters += r.total_teeth_affected
        total_ms += r.processing_time_ms
        severity_hist[r.highest_severity.value] += 1
        per_image.append({
            "image": fp.name, "modality": r.modality.value,
            "clusters": r.total_teeth_affected,
            "highest_severity": r.highest_severity.value,
            "ms": round(r.processing_time_ms, 1),
        })

    n_ok = len(files) - n_errors
    report = {
        "n_images": len(files),
        "n_ok": n_ok,
        "n_errors": n_errors,
        "mean_clusters_per_image": round(total_clusters / n_ok, 2) if n_ok else 0.0,
        "mean_ms_per_image": round(total_ms / n_ok, 1) if n_ok else 0.0,
        "severity_histogram": dict(severity_hist),
        "per_image": per_image,
    }
    return report
