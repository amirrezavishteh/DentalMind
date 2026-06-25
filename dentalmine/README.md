# DentalMind (Phase 1)

Dental X-ray AI pipeline. **Phase 1** implements the panoramic (OPG) modality
end-to-end with the full *integration* layer — per-tooth clustering (MadClip C3)
and ranked treatment-prompt generation (MadClip C4) — which is the part that
turns raw detections into something a dentist and patient can actually act on.

```
image → quality gate → modality router → preprocess → panoramic head
      → postprocess (NMS / FDI / surface / calibration)
      → C3 per-tooth clustering → C4 treatment prompts
      → annotated overlay + findings.json + treatment card + summary
```

## What's implemented (Phase 1)

| Stage | Module | Status |
|---|---|---|
| Schemas | `schema/` | ✅ |
| Quality gate (SNR/entropy/exposure) | `data/quality/quality_gate.py` | ✅ |
| Med-CLIP (MMKD → BiomedCLIP → Stub) | `models/medclip/` | ✅ |
| Modality router | `pipeline/router.py` | ✅ |
| Preprocessor (2D) | `pipeline/preprocessor.py` | ✅ |
| Panoramic head (2-stage) | `models/heads/panoramic_head.py` | ✅ |
| C3 clustering | `models/madclip/c3_clustering.py` | ✅ |
| C4 prompts (Tier 1 templates) | `models/madclip/c4_prompts.py` | ✅ |
| Postprocessor | `pipeline/postprocessor.py` | ✅ |
| Overlay renderer (2D) | `pipeline/overlay_renderer.py` | ✅ |
| Inference engine | `pipeline/inference_engine.py` | ✅ |
| CLI `infer` | `cli.py` | ✅ |
| Tests (CPU-only) | `tests/` | ✅ 11 passing |

**Deferred to later phases** (run on the A100): bitewing/periapical/CBCT heads,
C1 slice attention, C2 consistency filter, nnU-Net 3D segmentation, DentVFM
backbone, all training phases, C4 Tier-2 VLM (Qwen2-VL), serve/train/eval CLI.

## Med-CLIP fallback chain

`MedCLIPFactory` selects, in order:
1. **MMKD-CLIP** — if `weights/MMKD_B16.pth` exists (PRIMARY).
2. **BiomedCLIP** — auto-downloads from HuggingFace (~430 MB) — the active path.
3. **Stub** — deterministic offline encoder so the pipeline runs with no network
   (used by CI tests). Set `DENTALMIND_MEDCLIP=stub` to force it.

> The Stub produces *pseudo-random* pathology scores — it validates the
> architecture/integration, not detection accuracy. Use BiomedCLIP or MMKD-CLIP
> for real findings.

## Stage-1 tooth detection

With no DENTEX-trained YOLO checkpoint, the panoramic head emits an
anatomically-plausible 32-tooth FDI grid so C3/C4 are fully exercised. Provide a
trained checkpoint via `inference.panoramic_stage1_weights` in the config to use
real YOLOv11 tooth detection. (TODO: train on DENTEX.)

## Setup

```bash
cd dentalmine
pip install -e .
# real Med-CLIP + Stage-1 deps (optional locally; required for real findings):
pip install open_clip_torch ultralytics pydicom
```

## Run

```bash
# Offline / deterministic (stub encoder):
DENTALMIND_MEDCLIP=stub dentalmine infer \
  --input tests/fixtures/test_opg.jpg --modality opg --output ./results/ --device cpu

# Real BiomedCLIP (downloads weights on first use):
dentalmine infer --input <opg.jpg> --modality opg --output ./results/
```

Outputs in `--output`: `findings.json`, `original.png`, `annotated.png`,
`treatment_card.txt`, `summary.txt`.

## Test

```bash
DENTALMIND_MEDCLIP=stub pytest tests/ -v
```

## Datasets

See `../DATASETS.md` for the organized dataset tree (DENTEX, Tufts, Zenodo
caries, Mendeley mandible) under `../data/`.
