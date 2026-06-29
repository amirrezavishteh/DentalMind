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
        thr = float(inf.get("detection_score_threshold", 0.35))
        self.panoramic_head = PanoramicHead(
            medclip=self.medclip,
            stage1_weights=inf.get("panoramic_stage1_weights", None),
            stage1_model=inf.get("panoramic_stage1_model", "yolo11m.pt"),
            score_threshold=thr,
            device=self.device,
        )
        # BW / PA heads (zero-shot region detectors; modality-appropriate classes)
        from models.heads.bitewing_head import BitewingHead
        from models.heads.periapical_head import PeriapicalHead
        self.bitewing_head = BitewingHead(self.medclip, score_threshold=thr, device=self.device)
        self.periapical_head = PeriapicalHead(self.medclip, score_threshold=thr, device=self.device)
        self.heads_2d = {
            ModalityType.OPG: self.panoramic_head,
            ModalityType.BW: self.bitewing_head,
            ModalityType.PA: self.periapical_head,
        }
        self.postprocessor = PostProcessor.from_config(config)
        self.clusterer = FindingClusterer()
        # 3D CBCT path: per-slice detector + C2 neighbour-consistency decision.
        from models.heads.cbct_head import CBCTHead
        from models.madclip.c2_consistency import ConsistencyFilter
        self.cbct_head = CBCTHead(
            medclip=self.medclip,
            score_threshold=float(inf.get("detection_score_threshold", 0.35)),
            device=self.device,
        )
        self.c2 = ConsistencyFilter.from_config(config)
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

        # 3D CBCT path: a volume file (.nii/.nii.gz) or a DICOM-series directory.
        if _is_cbct_input(input_path, forced):
            return self._infer_cbct(input_path, patient_context, use_vlm, output_dir, t0, audit)

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

            head = self.heads_2d.get(modality_resolved, self.panoramic_head)
            raw_dets = head.detect(display_bgr)
            audit.append({"step": "detect", "head": type(head).__name__,
                          "raw_detections": len(raw_dets)})

            # FDI tooth grid only exists for the panoramic head; BW/PA cluster by
            # spatial proximity (DBSCAN) since they carry no per-tooth FDI map.
            fdi_map = head._stage1(display_bgr) if modality_resolved == ModalityType.OPG else None
            dets = self.postprocessor.process(
                raw_dets, modality_resolved, fdi_map=fdi_map,
                image_height_px=disp_h, audit_log=audit,
            )

            fdi_conf = 1.0 if modality_resolved == ModalityType.OPG else 0.0
            clusters = self.clusterer.cluster(dets, image_height_px=disp_h, fdi_confidence=fdi_conf)
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

    # ---- 3D CBCT pipeline ---------------------------------------------
    def _infer_cbct(self, input_path, patient_context, use_vlm, output_dir, t0, audit):
        """CBCT flow: slices -> CLIP per-slice detect -> C2 neighbour-consistency
        decision -> C3 cluster across slices -> C4 prompts.

        The 3D reasoning ("see neighbours and decide") is C2: a per-slice finding
        survives only if the same finding appears on enough physically-adjacent
        slices. C1's learned cross-slice fusion is the training-time counterpart
        (training/baseline_clip_2d3d.py / phase2) consumed by a trained 2.5D head.
        """
        try:
            from data.loaders.cbct_loader import load_cbct
            vol = load_cbct(input_path)
            audit.append({"step": "load_cbct", "n_slices": len(vol.slices),
                          "spacing_mm": round(vol.spacing_mm, 3)})

            display_slices, spacing = self.preprocessor.preprocess_cbct(vol.slices, vol.spacing_mm)

            # CLIP per-slice detection -> {slice_idx: [Detection]}
            per_slice = self.cbct_head.detect_volume(display_slices)
            raw_count = sum(len(v) for v in per_slice.values())
            audit.append({"step": "detect_per_slice", "raw_detections": raw_count})

            # C2: SEE NEIGHBOURS + DECIDE (neighbour-consistency vote)
            kept_per_slice = self.c2.filter(per_slice, spacing_mm=spacing)
            kept = [d for dets in kept_per_slice.values() for d in dets]
            audit.append({"step": "c2_consistency",
                          "kept": len(kept), "removed": raw_count - len(kept),
                          "min_support": self.c2.min_support})

            # calibrate confidences, then cluster across slices (DBSCAN fallback,
            # since CBCT detections carry no FDI tooth id)
            kept = self.postprocessor.calibrate_confidence(kept)
            clusters = self.clusterer.cluster(kept, fdi_confidence=0.0)
            audit.append({"step": "cluster", "n_clusters": len(clusters)})

            prompts = self.c4.generate(clusters, patient_context, use_vlm=use_vlm,
                                       image_path=input_path)

            # render a representative (middle) slice overlay
            mid = len(display_slices) // 2
            annotated = self.renderer.render_2d(display_slices[mid], clusters)

            result = InferenceResult(
                modality=ModalityType.CBCT, image_paths=[input_path], clusters=clusters,
                model_versions={**self.model_versions, "cbct": "clip-zeroshot+c2"},
                audit_log=audit,
            ).recompute_summary()
            result.processing_time_ms = (time.perf_counter() - t0) * 1000.0

            if output_dir is not None:
                self.renderer.export(result, clusters, prompts,
                                     display_slices[mid], annotated, output_dir)
            return result

        except (QualityError, FileNotFoundError, ImportError) as e:
            result = InferenceResult(
                modality=ModalityType.CBCT, image_paths=[input_path], error=str(e),
                model_versions=self.model_versions, audit_log=audit,
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


def _is_cbct_input(input_path, forced) -> bool:
    from pathlib import Path
    if forced == ModalityType.CBCT:
        return True
    p = Path(input_path)
    name = p.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return True
    if p.is_dir():
        # DICOM-series directory (has .dcm files, no flat 2D images at top level)
        return any(f.suffix.lower() in {".dcm", ".dicom"} for f in p.glob("*"))
    return False
