"""InferenceEngine — orchestrates the DentalMind pipeline.

Phase 1 implements the 2D / OPG path end-to-end. Sequence modalities
(CBCT/FMS/SERIAL_BW) and their C1/C2/nnU-Net stages raise NotImplementedError
until later phases land.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from rich.console import Console

from data.loaders.image_loader import ImageLoader
from data.quality.quality_gate import QualityError, QualityGate
from models.madclip.c3_clustering import FindingClusterer
from models.madclip.c4_prompts import TreatmentPromptGenerator
from models.medclip.medclip_factory import MedCLIPFactory
from models.heads.panoramic_head import PanoramicHead
from pipeline.overlay_renderer import OverlayRenderer
from pipeline.postprocessor import PostProcessor
from pipeline.preprocessor import Preprocessor
from pipeline.router import ModalityRouter
from schema.finding import InferenceResult, ModalityType, parse_modality
from schema.patient import PatientContext

_console = Console()

SEQUENCE_MODALITIES = {ModalityType.CBCT, ModalityType.FMS, ModalityType.SERIAL_BW}

MODEL_VERSIONS = {
    "pipeline": "dentalmind-0.1.0-phase1",
    "panoramic_stage1": "yolo-placeholder-grid",
    "panoramic_stage2": "medclip-zero-shot",
}


def _resolve_device(cfg) -> str:
    inf = cfg.get("inference", {}) if hasattr(cfg, "get") else {}
    dev = inf.get("device", "auto")
    if dev == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return dev


class InferenceEngine:
    def __init__(self, config, checkpoint_dir: Optional[str] = None, device: Optional[str] = None):
        self.config = config
        self.device = device or _resolve_device(config)

        inf = config.get("inference", {})
        data = config.get("data", {})
        rend = config.get("rendering", {})

        self.loader = ImageLoader()
        self.quality_gate = QualityGate.from_config(config)

        _console.print(f"[cyan]Loading Med-CLIP encoder on {self.device}...[/cyan]")
        self.medclip = MedCLIPFactory.get(config, device=self.device)

        self.router = ModalityRouter(medclip=self.medclip)
        self.preprocessor = Preprocessor(
            yolo_size=int(data.get("yolo_image_size", 640)),
            dentvfm_size=int(data.get("dentvfm_image_size", 518)),
        )
        self.panoramic_head = PanoramicHead(
            medclip=self.medclip,
            stage1_weights=inf.get("panoramic_stage1_weights", None),
            stage1_model=inf.get("panoramic_stage1_model", "yolo11m.pt"),
            score_threshold=float(inf.get("detection_score_threshold", 0.35)),
            device=self.device,
        )
        self.postprocessor = PostProcessor.from_config(config)
        self.clusterer = FindingClusterer()
        self.c4 = TreatmentPromptGenerator(medclip=self.medclip)
        self.renderer = OverlayRenderer(
            overlay_alpha=float(rend.get("overlay_alpha", 0.30)),
            show_fdi_labels=bool(rend.get("show_fdi_labels", True)),
        )
        self.model_versions = dict(MODEL_VERSIONS)
        self.model_versions["medclip"] = getattr(self.medclip, "name", "unknown")

    # ---- main ----------------------------------------------------------
    def infer(
        self,
        input_path: str,
        modality: ModalityType | str = "auto",
        patient_context: Optional[PatientContext] = None,
        use_vlm: bool = False,
        output_dir: Optional[str] = None,
    ) -> InferenceResult:
        t0 = time.perf_counter()
        audit: list = []
        forced = None if modality in ("auto", None) else parse_modality(modality)

        try:
            loaded = self.loader.load(input_path)
            audit.append({"step": "load", "n_images": len(loaded.images)})

            modality_resolved, mconf = self.router.classify(loaded, forced=forced)
            audit.append({"step": "route", "modality": modality_resolved.value, "confidence": mconf})

            if modality_resolved in SEQUENCE_MODALITIES:
                raise NotImplementedError(
                    f"Modality '{modality_resolved.value}' (sequence) is not implemented in "
                    f"Phase 1. Only 2D OPG/BW/PA are supported."
                )

            image = loaded.images[0]
            qreport = self.quality_gate.check_or_raise(image, loaded.dicom_meta)
            audit.append({"step": "quality", "snr_db": round(qreport.snr_db, 2),
                          "entropy_bits": round(qreport.entropy_bits, 2)})

            pre = self.preprocessor.preprocess_2d(image, modality_resolved, snr_db=qreport.snr_db)
            display_bgr = pre.display_bgr
            disp_h = display_bgr.shape[0]

            raw_dets = self.panoramic_head.detect(display_bgr)
            audit.append({"step": "detect", "raw_detections": len(raw_dets)})

            fdi_map = self.panoramic_head._stage1(display_bgr)
            dets = self.postprocessor.process(
                raw_dets, modality_resolved, fdi_map=fdi_map,
                image_height_px=disp_h, audit_log=audit,
            )

            clusters = self.clusterer.cluster(dets, image_height_px=disp_h)
            audit.append({"step": "cluster", "n_clusters": len(clusters)})

            annotated = self.renderer.render_2d(display_bgr, clusters)
            prompts = self.c4.generate(
                clusters, patient_context, use_vlm=use_vlm, image_path=input_path
            )

            result = InferenceResult(
                modality=modality_resolved,
                image_paths=loaded.paths,
                clusters=clusters,
                model_versions=self.model_versions,
                patient_id=None,
                audit_log=audit,
            ).recompute_summary()
            result.processing_time_ms = (time.perf_counter() - t0) * 1000.0

            if output_dir is not None:
                self.renderer.export(result, clusters, prompts, display_bgr, annotated, output_dir)
            return result

        except (QualityError, NotImplementedError, FileNotFoundError) as e:
            result = InferenceResult(
                modality=forced or ModalityType.OPG,
                image_paths=[input_path],
                error=str(e),
                model_versions=self.model_versions,
                audit_log=audit,
            )
            result.processing_time_ms = (time.perf_counter() - t0) * 1000.0
            if output_dir is not None:
                self._export_error(result, output_dir)
            return result

    def _export_error(self, result: InferenceResult, output_dir: str):
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "findings.json", "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))
        with open(out / "summary.txt", "w", encoding="utf-8") as fh:
            fh.write(result.to_report())
