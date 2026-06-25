You are building **DentalMind** — a production dental AI pipeline. Implement the complete project now. No clarifying questions. Where spec is ambiguous, pick the simpler option and leave a TODO comment.

---

## WHAT YOU ARE BUILDING

A CLI tool that takes dental X-ray images (Bitewing, Panoramic, Periapical, Full-Mouth Series, CBCT) and outputs:
1. Annotated images with color-coded pathology overlays
2. Structured JSON findings per tooth (FDI notation)
3. Ranked treatment prompt cards for the dentist

Core architecture:
- **Shared DentVFM-2D ViT-L encoder** for all 2D modalities (one backbone, multiple heads)
- **MadClip C1**: cross-slice attention adapter — gives 2D encoder 3D awareness for CBCT/FMS
- **MadClip C2**: neighbor consistency filter — removes false positives by requiring finding to persist across K adjacent slices
- **MadClip C3**: clusters per-detection outputs into compound per-tooth findings
- **MadClip C4**: generates ranked treatment prompts from clusters (Tier 1 = templates, Tier 2 = LLM)
- **nnU-Net v2** for full 3D CBCT volumetric segmentation (runs in parallel with 2.5D head)

---

## STACK

```
Python 3.11, PyTorch 2.3+CUDA 12.1
ultralytics>=8.2          # YOLOv8 / YOLOv11
timm>=1.0                 # ViT-L backbone (DentVFM fallback)
open_clip_torch            # MMKD-CLIP + BiomedCLIP (install via MMKD repo)
transformers>=4.40         # BiomedNLP-BiomedBERT tokenizer + Swin-T + Qwen2-VL-7B
nnunetv2                  # CBCT 3D segmentation
monai                     # 3D medical imaging utils
SimpleITK, pydicom        # DICOM + CBCT loading
opencv-python, Pillow
scikit-learn              # DBSCAN for C3 Strategy 2
vtk                       # 3D mesh rendering
typer, rich               # CLI
pydantic>=2.0             # schemas
fastapi, uvicorn          # serve mode
wandb                     # training logging
plastimatch               # DRR generation (install separately, CLI tool)
gdown                     # Google Drive download for MMKD-CLIP weights
pytest
```

### MMKD-CLIP installation (run once before anything else):
```bash
git clone https://github.com/wangshansong1/MMKD-CLIP.git external/MMKD-CLIP
cd external/MMKD-CLIP && pip install -r requirements.txt && cd ../..

# Download pretrained weights from Google Drive
pip install gdown
gdown 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR -O weights/MMKD_B16.pth
```

### Med-CLIP model hierarchy (use in this priority order):
```
1. MMKD-CLIP          — PRIMARY. Best overall on X-ray (2025). AVAILABLE NOW.
                        GitHub:  github.com/wangshansong1/MMKD-CLIP
                        Weights: Google Drive 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR (MMKD_B16.pth)
                        Model:   ViT-B-16-quickgelu  via open_clip create_model_and_transforms
                        Text:    microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
                        Distills 9 biomedical CLIPs. #1 AUC on X-ray across 58 datasets.
                        Evaluation dataset: huggingface.co/datasets/Shansong/zeroshotclassification

2. BiomedCLIP         — FALLBACK if MMKD weights unavailable.
                        HuggingFace: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
                        15M PMC image-text pairs, PubMedBERT text + ViT-B/16 image encoder.

3. BMCA-CLIP          — Best for zero-shot breadth (+24.67% over PMC-CLIP).
                        HuggingFace: mrclarke/bmca-clip (concept-filtered variant).
                        Use as auxiliary encoder for rare finding types.

4. RegionMed-CLIP     — Only model with explicit DENTAL category + ROI-level annotation.
                        arXiv: 2508.05244 — use for C4 Tier 2 grounding when released.
                        Dataset: MedRegion-500k covering 12 imaging categories incl. Dental.

5. Original MedCLIP   — Historical baseline, superseded; do NOT use as primary.
                        HuggingFace: zifengw/medclip
```

---

## PROJECT LAYOUT

Create exactly this structure:

```
dentalmine/
├── cli.py
├── config/
│   ├── default.yaml
│   └── modalities.yaml
├── data/
│   ├── loaders/
│   │   ├── dicom_loader.py
│   │   ├── cbct_loader.py
│   │   └── fms_loader.py
│   ├── augmentation/
│   │   ├── dental_augment.py
│   │   └── drr_generator.py
│   ├── datasets/
│   │   ├── dentex_dataset.py
│   │   ├── bitewing_dataset.py
│   │   ├── cbct_dataset.py
│   │   └── joint_dataset.py
│   └── quality/
│       └── quality_gate.py
├── models/
│   ├── backbone/
│   │   └── dentvfm.py
│   ├── medclip/
│   │   ├── mmkd_clip.py           # PRIMARY: MMKD-CLIP (ViT-B-16-quickgelu, MMKD_B16.pth)
│   │   ├── biomedclip.py          # FALLBACK: microsoft/BiomedCLIP if MMKD weights absent
│   │   ├── regionmed_clip.py      # DENTAL ROI: RegionMed-CLIP for grounding (stub+TODO)
│   │   └── medclip_factory.py     # factory: tries MMKD → BiomedCLIP → error
│   ├── madclip/
│   │   ├── c1_slice_attention.py
│   │   ├── c2_consistency.py
│   │   ├── c3_clustering.py
│   │   └── c4_prompts.py
│   ├── heads/
│   │   ├── bitewing_head.py
│   │   ├── panoramic_head.py
│   │   ├── periapical_head.py
│   │   └── cbct_head.py
│   ├── segmentation/
│   │   └── nnunet_wrapper.py
│   └── vlm/
│       └── dentvlm.py
├── pipeline/
│   ├── router.py
│   ├── preprocessor.py
│   ├── inference_engine.py
│   ├── postprocessor.py
│   └── overlay_renderer.py
├── training/
│   ├── phase1_pretrain.py
│   ├── phase2_c1_adapter.py
│   ├── phase3_multitask.py
│   ├── phase4_distillation.py
│   └── trainer_base.py
├── schema/
│   ├── finding.py
│   ├── prompt.py
│   └── patient.py
├── templates/
│   └── treatment_templates.yaml
├── scripts/
│   ├── download_dentex.py
│   ├── download_weights.py
│   ├── extract_cbct_slices.py
│   └── generate_drr.py
├── tests/
│   ├── test_pipeline_e2e.py
│   ├── test_madclip.py
│   ├── test_schemas.py
│   └── fixtures/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## CLI — ALL COMMANDS MUST WORK

```bash
# Inference
dentalmine infer \
  --input path/to/image_or_dir \
  --output ./results/ \
  --modality auto \          # auto | bw | opg | pa | fms | cbct
  --use-vlm \                # enable DentVLM Tier 2 (optional)
  --patient-age 45 \
  --checkpoint ./ckpt/best.pt

# Batch inference
dentalmine infer-batch \
  --input-dir ./patients/ \
  --output-dir ./results/ \
  --workers 4

# Training phases (run in order)
dentalmine train phase1 \
  --data-dir ./data/ \
  --epochs 100 \
  --fast-dev-run            # 1-batch smoke test

dentalmine train phase2 \
  --cbct-dir ./data/cbct/ \
  --backbone-ckpt ./ckpt/phase1.pt \
  --epochs 50

dentalmine train phase3 \
  --config ./config/default.yaml \
  --resume \
  --fast-dev-run

dentalmine train phase4 \
  --cbct-dir ./data/cbct/ \
  --pseudo-output ./data/pseudo/

# Data utilities
dentalmine data download-dentex --output ./data/dentex/
dentalmine data download-weights --output ./weights/
dentalmine data extract-slices \
  --cbct-dir ./data/cbct/ \
  --output ./data/cbct_slices/
dentalmine data generate-drr \
  --cbct-dir ./data/cbct/ \
  --output ./data/drr/ \
  --mode panoramic           # panoramic | periapical
dentalmine data quality-check \
  --input-dir ./raw/ \
  --report quality.json

# Evaluation
dentalmine eval \
  --checkpoint ./ckpt/best.pt \
  --test-dir ./test/ \
  --modality opg

# API server
dentalmine serve \
  --host 0.0.0.0 \
  --port 8080 \
  --checkpoint ./ckpt/best.pt
```

---

## SCHEMAS (implement first — everything depends on these)

### `schema/finding.py`
```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Urgency(str, Enum):
    IMMEDIATE = "immediate_24h"
    SOON = "soon_2_weeks"
    ROUTINE = "routine_next_visit"
    MONITOR = "monitor_6_months"

class ModalityType(str, Enum):
    BW = "bitewing"
    SERIAL_BW = "serial_bitewing"
    OPG = "panoramic"
    PA = "periapical"
    FMS = "full_mouth_series"
    CBCT = "cbct"

class Detection(BaseModel):
    class_name: str
    confidence: float
    calibrated_confidence: float
    bbox_xyxy: List[float]          # [x1,y1,x2,y2] normalized 0-1
    mask_rle: Optional[str] = None  # RLE encoded mask
    surface: Optional[str] = None   # mesial/distal/occlusal/cervical/buccal
    slice_idx: Optional[int] = None # CBCT slice index
    flagged_for_review: bool = False

class ToothCluster(BaseModel):
    tooth_fdi: str                   # "26", "unknown_1"
    severity: Severity
    urgency: Urgency
    pattern: str                     # key from PATTERN_LIBRARY
    findings: List[Detection]
    cluster_confidence: float
    surfaces_affected: List[str]
    bone_loss_mm: Optional[float] = None
    has_3d_mask: bool = False
    neighbor_support_score: float = 0.0  # C2 output

class InferenceResult(BaseModel):
    patient_id: Optional[str] = None
    modality: ModalityType
    image_paths: List[str]
    clusters: List[ToothCluster]
    total_teeth_affected: int
    highest_severity: Severity
    processing_time_ms: float
    model_versions: dict
    audit_log: List[dict] = Field(default_factory=list)

    def to_report(self) -> str:
        """Human-readable summary for treatment_card.txt"""
        ...
```

### `schema/prompt.py`
```python
class TreatmentOption(BaseModel):
    rank: int
    text: str
    when_to_use: str

class PromptOutput(BaseModel):
    tooth_fdi: str
    pattern: str
    options: List[TreatmentOption]
    tests_suggested: List[str]
    urgency: Urgency
    red_flags: List[str]
    tier: str   # "template" | "vlm_advisory"
    disclaimer: str = (
        "⚠ AI second opinion only. "
        "Final diagnosis and treatment plan subject to clinical judgment."
    )
```

### `schema/patient.py`
```python
class PatientContext(BaseModel):
    age: Optional[int] = None
    smoker: Optional[bool] = None
    diabetic: Optional[bool] = None
    last_visit_months: Optional[int] = None
    existing_restorations: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    chief_complaint: Optional[str] = None
```

---

## STAGE IMPLEMENTATIONS

### `data/quality/quality_gate.py`
```python
# QualityGate class
# check(image: np.ndarray) -> QualityReport(snr_db, entropy_bits, passed, reason)
# Reject if: SNR < 15dB OR entropy < 4.5 bits OR >40% pixels outside [10,245]
# SNR: signal_power / noise_power where noise estimated from corner patches
# Entropy: scipy.stats.entropy(histogram/histogram.sum())
# DICOM: extract modality_hint, slice_spacing, kVp, mAs from tags
# Raise QualityError with clear message if image rejected
```

### `pipeline/router.py`
```python
# ModalityRouter class
# Primary path: read DICOM tag (0008,0060)
#   - "DX" + single image → check aspect ratio:
#       wide (W>2H): OPG
#       square/tall, small FoV: PA or BW (use tooth count heuristic)
#   - "CT" or 3D DICOM series with ImagePositionPatient → CBCT
# Secondary path (no DICOM): MMKD-CLIP zero-shot classification
#   Load MMKDCLIPEncoder from models/medclip/mmkd_clip.py
#   Text templates:
#     ["bitewing dental X-ray showing interproximal teeth",
#      "panoramic dental X-ray showing full mouth",
#      "periapical X-ray showing tooth root and apex",
#      "dental CBCT cone beam CT cross-section",
#      "full mouth series dental radiograph"]
#   Run zero-shot similarity → pick highest score → map to ModalityType
#   Fallback to BiomedCLIPEncoder if MMKD weights not found
# FMS detection: folder with 14-18 images all classified as PA → FMS
# Returns: ModalityType + confidence float
```

### `models/backbone/dentvfm.py`
```python
# DentVFMEncoder class wrapping ViT-L/14
#
# Priority order for weights:
#   1. DentVFM pretrained weights (if path provided in config)
#   2. timm ViT-L/14 with DINOv2 weights:
#      timm.create_model('vit_large_patch14_dinov2', pretrained=True)
#   3. MMKD-CLIP visual encoder (RECOMMENDED fallback — distills 9 biomedical CLIPs):
#      from open_clip import create_model_and_transforms, get_mean_std
#      mean, std = get_mean_std()
#      model, _, preprocess = create_model_and_transforms(
#          model_name='ViT-B-16-quickgelu',
#          pretrained='./weights/MMKD_B16.pth',
#          precision='amp', device=device,
#          force_quick_gelu=True, mean=mean, std=std, inmem=True,
#          text_encoder_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
#      use model.encode_image as the image encoder
#      Note: MMKD-CLIP is ViT-B/16 (D=512, patch=16, input=224)
#            vs DentVFM ViT-L/14 (D=1024, patch=14, input=518)
#      Add Linear(512→1024) projection layer after MMKD encoder
#      to match the rest of the pipeline's expected D=1024
#   4. BiomedCLIP ViT-B/16 (if MMKD weights not found):
#      from open_clip import create_model_from_pretrained
#      model, _ = create_model_from_pretrained(
#          'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#   5. timm ViT-L/14 ImageNet (last resort)
#
# encode(x: Tensor[B,3,518,518]) -> Tensor[B, 1369, 1024]   (DentVFM path)
# encode(x: Tensor[B,3,224,224]) -> Tensor[B, 196, 512→1024] (MMKD-CLIP path)
#   auto-detect which encoder is loaded and adjust input size accordingly
# encode_spatial(x) -> Tensor[B, H', W', D]
# encode_cls(x) -> Tensor[B, D]
#
# FREEZE weights by default (encoder.requires_grad_(False))
# Only unfreeze during Phase 1 training
```

### `models/medclip/mmkd_clip.py` — PRIMARY Med-CLIP (implement fully)
```python
# MMKDCLIPEncoder — PRIMARY Med-CLIP for the entire pipeline
#
# Loads MMKD-CLIP (ViT-B-16-quickgelu) from local weights MMKD_B16.pth
# Source: github.com/wangshansong1/MMKD-CLIP
# Weights: Google Drive gdown 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR -O weights/MMKD_B16.pth
#
# Exact loading code from the official README:
import sys; sys.path.insert(0, 'external/MMKD-CLIP/src')
from open_clip import create_model_and_transforms, get_mean_std, HFTokenizer

class MMKDCLIPEncoder:
    TEXT_ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    MODEL_NAME   = "ViT-B-16-quickgelu"
    CONTEXT_LEN  = 256

    def __init__(self, weights_path: str = "./weights/MMKD_B16.pth", device: str = "cuda"):
        mean, std = get_mean_std()
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=self.MODEL_NAME,
            pretrained=weights_path,
            precision="amp",
            device=device,
            force_quick_gelu=True,
            mean=mean, std=std,
            inmem=True,
            text_encoder_name=self.TEXT_ENCODER,
        )
        self.tokenizer = HFTokenizer(
            self.TEXT_ENCODER,
            context_length=self.CONTEXT_LEN,
        )
        self.model.eval()
        self.device = device

    def encode_image(self, images: "Tensor") -> "Tensor":
        # images: [B, 3, H, W] — already preprocessed with self.preprocess
        with torch.no_grad(), torch.autocast("cuda"):
            feats = self.model.encode_image(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, text_prompts: list[str]) -> "Tensor":
        texts = torch.cat(
            [self.tokenizer(t).to(self.device) for t in text_prompts], dim=0
        )
        with torch.no_grad(), torch.autocast("cuda"):
            feats = self.model.encode_text(texts)
        return feats / feats.norm(dim=-1, keepdim=True)

    def zero_shot(self, image: "Tensor", text_prompts: list[str]) -> dict:
        # Returns {prompt: probability} sorted descending
        img_f = self.encode_image(image)
        txt_f = self.encode_text(text_prompts)
        logits = (self.model.logit_scale.exp() * img_f @ txt_f.t()).softmax(dim=-1)
        return {p: float(logits[0, i]) for i, p in enumerate(text_prompts)}

    def dental_zero_shot(self, image: "Tensor") -> dict:
        # Dental-specific prompt set for pathology grounding
        prompts = [
            "dental caries interproximal on bitewing radiograph",
            "dentin caries near pulp on dental X-ray",
            "periapical lesion on endodontic radiograph",
            "alveolar bone loss on dental radiograph",
            "impacted tooth on panoramic X-ray",
            "calculus deposit on dental radiograph",
            "healthy tooth on dental radiograph",
        ]
        return self.zero_shot(image, prompts)

    @classmethod
    def available(cls, weights_path: str = "./weights/MMKD_B16.pth") -> bool:
        import os
        return os.path.exists(weights_path)
```

### `models/medclip/biomedclip.py` — FALLBACK only
```python
# BiomedCLIPEncoder — used ONLY when MMKD_B16.pth is not present
# Auto-downloads from HuggingFace on first use (~430 MB)
# API is identical to MMKDCLIPEncoder so factory can swap transparently
#
# from open_clip import create_model_from_pretrained, get_tokenizer
# HF_ID = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
# model, preprocess = create_model_from_pretrained(f'hf-hub:{HF_ID}')
# tokenizer = get_tokenizer(f'hf-hub:{HF_ID}')
#
# Implement same encode_image / encode_text / zero_shot / dental_zero_shot API
# Add Linear(512→1024) projection in encode_image output (to match pipeline D=1024)
```

### `models/medclip/medclip_factory.py`
```python
# MedCLIPFactory.get(config, device) -> MMKDCLIPEncoder | BiomedCLIPEncoder
#
# Priority:
#   1. If config.mmkd_weights and MMKDCLIPEncoder.available(config.mmkd_weights):
#        return MMKDCLIPEncoder(config.mmkd_weights, device)
#   2. Else:
#        log warning "MMKD_B16.pth not found — falling back to BiomedCLIP"
#        return BiomedCLIPEncoder(device)
#   3. After loading, print which model was loaded + weights path
```

---

### `models/madclip/c1_slice_attention.py`
```python
# CrossSliceAttentionAdapter — ~2M params, sits on top of frozen DentVFM
#
# Architecture:
#   Input: List of K+1 feature maps, each [B, 37, 37, 1024]
#   Flatten spatial: [B, 1369, 1024] per slice
#   Stack: [B, K+1, 1369, 1024]
#   Reshape for attention: [B*(K+1), 1369, 1024]
#   Multi-head cross-attention (query=central slice, kv=all slices):
#     nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
#     query: [B, 1369, 1024]
#     key, value: [B, (K+1)*1369, 1024]
#   Output projection: Linear(1024→1024) + LayerNorm
#   Reshape back: [B, 37, 37, 1024]
#   Residual: output + central_features
#
# forward(features_list: List[Tensor], target_idx: int) -> Tensor[B,37,37,1024]
#
# K selection (called by inference engine):
#   CBCT: K = max(2, round(1.0 / slice_spacing_mm))
#   FMS:  K = 1
#   SERIAL_BW: K = 2
#   Single 2D: return features_list[0] unchanged (identity)
#
# Boundary handling: zero-pad missing neighbors
```

### `models/madclip/c2_consistency.py`
```python
# ConsistencyFilter — pure inference-time, no training needed
#
# filter(
#   detections_per_image: Dict[int, List[Detection]],
#   spacing_mm: float = 1.0,
#   min_support: int = 3,
#   iou_threshold: float = 0.40,
# ) -> Dict[int, List[Detection]]
#
# Algorithm for each detection D at image index N:
#   neighbors = images within physical distance 1.0mm from N
#   support = count of neighbors having detection with:
#     - same class_name
#     - IoU(bbox_D, bbox_neighbor) >= iou_threshold
#   if support >= min_support: keep D, boost confidence by 1 + 0.08*support
#   else: discard D
#
# Also implement back_project_3d_filter() as alternative:
#   Project all detections to 3D voxel space using slice positions
#   Run scipy.ndimage.label() for connected components
#   Discard components with volume < 8mm³ (noise threshold)
#   Re-project surviving components back to per-slice detections
#
# Single 2D images (no sequence): return input unchanged
```

### `models/madclip/c3_clustering.py`
```python
# FindingClusterer
#
# PATTERN_LIBRARY = {
#   'caries_enamel':        {'severity': 'LOW',      'urgency': 'MONITOR'},
#   'caries_dentin':        {'severity': 'HIGH',     'urgency': 'SOON'},
#   'caries_pulp_risk':     {'severity': 'CRITICAL', 'urgency': 'IMMEDIATE'},
#   'caries_with_bone':     {'severity': 'HIGH',     'urgency': 'SOON'},
#   'caries_with_pal':      {'severity': 'CRITICAL', 'urgency': 'IMMEDIATE'},
#   'bone_loss_only':       {'severity': 'MEDIUM',   'urgency': 'ROUTINE'},
#   'multi_tooth_bone':     {'severity': 'HIGH',     'urgency': 'SOON'},
#   'impaction':            {'severity': 'MEDIUM',   'urgency': 'ROUTINE'},
#   'endo_complex':         {'severity': 'CRITICAL', 'urgency': 'IMMEDIATE'},
#   'calculus_only':        {'severity': 'LOW',      'urgency': 'ROUTINE'},
#   'restoration_issue':    {'severity': 'MEDIUM',   'urgency': 'ROUTINE'},
# }
#
# Strategy 1 (default — rule-based, no training):
# cluster_by_fdi(detections: List[Detection]) -> List[ToothCluster]
#   Group by detection.tooth_fdi
#   For each group: determine pattern from class_names present
#   Assign severity/urgency from PATTERN_LIBRARY
#   Assign surface_affected from detection.surface values
#   Extract bone_loss_mm from 'bone_loss' detections bbox height * px_spacing
#
# Strategy 2 (fallback when FDI confidence < 0.65):
# cluster_by_dbscan(detections: List[Detection]) -> List[ToothCluster]
#   Features: [x_center_norm, y_center_norm, class_embedding(64d)]
#   DBSCAN(eps=0.12, min_samples=1)
#   Assign pattern from majority class in cluster
#
# merge_with_3d(clusters_2d, nnunet_output) -> List[ToothCluster]
#   Match 2D clusters to 3D tooth instances by FDI
#   Add has_3d_mask=True and cbct_3d metadata
```

### `models/madclip/c4_prompts.py`
```python
# TreatmentPromptGenerator
#
# Tier 1: generate_from_template(cluster, patient_context) -> PromptOutput
#   Load treatment_templates.yaml
#   Match cluster.pattern → template entry
#   Fill: {tooth}, {surface}, {bone_mm}, {age}, {urgency_text}
#   Append disclaimer always
#   Use for all cases by default — safe for clinical deployment
#
# Tier 2: generate_from_vlm(cluster, image_path, patient_context) -> PromptOutput
#   TWO-STEP pipeline:
#
#   Step A — MMKD-CLIP visual grounding (always runs in Tier 2):
#     Load MMKDCLIPEncoder from models/medclip/mmkd_clip.py
#     For each finding in cluster: encode ROI image → embedding
#     Call mmkd.dental_zero_shot(roi_image) to get probability per pathology
#     Use top-3 scoring descriptors as grounding context for Step B
#     MMKD-CLIP is preferred here because it distills 9 CLIP models
#     and achieves #1 AUC on X-ray — most reliable grounding signal
#
#   Step B — Qwen2-VL-7B text generation (or DentVLM if available):
#     System prompt:
#       "You are a dental clinical decision support assistant.
#        Given structured AI detection results and MMKD-CLIP visual grounding context,
#        output ONLY valid JSON matching this schema: {schema}.
#        Use hedged language: 'suggests', 'consider', 'possible'.
#        Never give a definitive diagnosis."
#     User content:
#       - cluster JSON (tooth, severity, pattern, findings)
#       - patient_context JSON (age, history)
#       - MMKD-CLIP dental_zero_shot scores from Step A
#       - original image (if Qwen2-VL-7B; image-text model)
#     Parse response → PromptOutput
#     Label tier='vlm_advisory'
#
#   If RegionMed-CLIP is available (arXiv 2508.05244):
#     Replace MMKD-CLIP grounding with RegionMed-CLIP
#     It has explicit DENTAL category + ROI-level annotations
#     Mark TODO: "Replace with RegionMed-CLIP when weights released"
#
#   Activate only when --use-vlm flag set
#   Always append disclaimer regardless of tier
#
# treatment_templates.yaml must cover all 11 patterns in PATTERN_LIBRARY
# Each template has: options (list, ranked), tests_suggested, red_flags
```

### `models/heads/bitewing_head.py`
```python
# YOLOv8x-seg with DentVFM feature injection
#
# 8 classes:
#   0:enamel_caries  1:dentin_caries  2:secondary_caries
#   3:bone_loss      4:furcation_defect  5:calculus
#   6:overhanging_filling  7:open_contact
#
# Implementation:
#   Load ultralytics YOLOv8x-seg (ultralytics.YOLO('yolov8x-seg.yaml'))
#   In forward: use DentVFM spatial features [B,37,37,1024] as backbone output
#   Add adapter conv layers to match YOLO's expected feature dimensions:
#     Conv2d(1024→512, 1x1) → P3 features
#     Conv2d(1024→256, 1x1) → P4 features (downsampled 2x)
#     Conv2d(1024→128, 1x1) → P5 features (downsampled 4x)
#   Feed into YOLO neck + head
#
# detect(image_tensor, features=None) -> List[Detection]
#   Returns post-NMS detections with mask_rle when segmentation head fires
```

### `models/heads/panoramic_head.py`
```python
# TWO-STAGE pipeline
#
# Stage 1 — Tooth detection + FDI enumeration:
#   Model: YOLOv11 (ultralytics YOLO('yolo11m.yaml'))
#   32 classes: FDI notation 11-18, 21-28, 31-38, 41-48
#   Returns: Dict[fdi_str, Dict{'bbox','confidence'}]
#   Trained on DENTEX dataset tooth annotations
#
# Stage 2 — Per-tooth pathology (MMKD-CLIP zero-shot + HierarchicalDet):
#   DEFAULT (zero-shot, no training needed):
#     Load MMKDCLIPEncoder from models/medclip/mmkd_clip.py
#     For each tooth ROI (cropped 224×224 and preprocessed with mmkd.preprocess):
#       image_features = mmkd.encode_image(roi_tensor)
#       text_prompts = [
#           "dental caries on tooth X-ray",
#           "deep caries near pulp on dental radiograph",
#           "periapical lesion on dental X-ray",
#           "impacted tooth on panoramic X-ray",
#           "healthy tooth on dental radiograph",
#       ]
#       scores = mmkd.zero_shot(roi_tensor, text_prompts)
#       Threshold each pathology class at 0.35
#     Returns: Dict[fdi_str, List[str]] (pathologies per tooth)
#
#   FINE-TUNED (when labeled data available):
#     Use HierarchicalDet from github.com/ibrahimethemhamamci/HierarchicalDet
#     Replace MMKD-CLIP zero-shot with supervised multi-label classifier
#     Mark TODO until HierarchicalDet is cloned into external/
#
# Combine Stage 1 + Stage 2 → List[Detection] with tooth_fdi field set
```

### `models/heads/periapical_head.py`
```python
# YOLOv8s + Swin-T Transformer hybrid
#
# Architecture:
#   Backbone: replace YOLOv8s stages 3-5 with Swin-T (tiny) from timm
#   timm.create_model('swin_tiny_patch4_window7_224', features_only=True)
#   Keep YOLO neck (PANet) + detection head
#
# Classes:
#   periapical_lesion, caries, bone_level_reduced,
#   root_filling_present, root_filling_inadequate,
#   resorption, instrument_separation
#
# detect(image_tensor) -> List[Detection]
```

### `models/heads/cbct_head.py`
```python
# Lightweight 2.5D detection head on C1-enriched features
# Input: [B, 37, 37, 1024] (C1 output)
# Architecture:
#   Conv2d(1024→256, 3x3, padding=1) + BN + ReLU
#   Conv2d(256→128, 3x3, padding=1) + BN + ReLU
#   Detection head: Conv2d(128→num_classes*5, 1x1)
#   Output: [B, 37, 37, num_classes*5] (class + bbox offsets)
# Classes: tooth_present, periapical_lesion, bone_defect,
#          calcification, nerve_proximity, sinus_proximity
# decode_predictions(raw) -> List[Detection]
```

### `models/segmentation/nnunet_wrapper.py`
```python
# Wraps nnU-Net v2 for full 3D segmentation
# Pre-trained weights: DentalSegmentator from Zenodo (zenodo.org/records/10829675)
# Labels: 0=bg, 1=upper_skull, 2=mandible, 3=upper_teeth, 4=lower_teeth, 5=mandibular_canal
#
# Run via subprocess to isolate memory:
#   nnUNetv2_predict -i input_dir -o output_dir -d DATASET_ID -c 3d_fullres
#
# segment(volume_path: str, output_dir: str) -> segmentation_path: str
# get_tooth_instances(seg_path) -> Dict[fdi_str, np.ndarray]
#   Connected components on labels 3+4
#   Assign FDI by centroid: x→left/right, z→upper/lower, y→anterior/posterior
```

### `pipeline/preprocessor.py`
```python
# Preprocessor class
#
# preprocess_2d(image: np.ndarray, modality: ModalityType) -> Tensor:
#   1. Convert to grayscale if needed, then 3-channel (repeat)
#   2. CLAHE: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply()
#   3. Foreground crop: Otsu threshold → bounding box + 10px pad
#   4. Normalize: (img / 255.0 - 0.485) / 0.229  (ImageNet mean/std)
#   5. Resize: 640×640 (YOLO) and 518×518 (DentVFM) as separate outputs
#   6. PA only: if quality_report.snr_db < 20: apply lightweight denoising
#      (cv2.fastNlMeansDenoising as ESRGAN placeholder, TODO: replace)
#
# preprocess_cbct(volume_path: str) -> (List[Tensor], slice_spacing_mm: float):
#   1. Load with SimpleITK: sitk.ReadImage(volume_path)
#   2. Resample to 0.4mm isotropic: sitk.ResampleImageFilter
#   3. Z-score normalize: (volume - mean) / std per volume
#   4. Extract axial slices → list of [1, H, W] tensors
#   5. Return (slices, physical_spacing)
#
# preprocess_fms(image_paths: List[str]) -> Dict[str, Tensor]:
#   1. Run preprocess_2d on each
#   2. Spatial ordering: assign to FDI quadrant using:
#      - Image filename hints (LR, UR, LL, UL in filename)
#      - Fallback: manual quadrant assignment (1-4) by file order
#   3. Return {quadrant_str: tensor}
```

### `pipeline/postprocessor.py`
```python
# PostProcessor class
#
# process(raw_detections, modality, fdi_map=None) -> List[Detection]:
#   1. nms(detections, iou_thresh=0.45) — per-class NMS
#   2. score_filter(detections, threshold=0.35)
#   3. assign_fdi(detections, fdi_map, modality)
#   4. assign_surface(detections)
#   5. calibrate_confidence(detections, temperature=1.3)
#   6. flag_review = [d for d if d.calibrated_confidence < 0.55]
#   7. audit_log.append({'step':'postprocess', 'flagged_count': len(flag_review)})
#
# assign_fdi rules:
#   OPG: use Stage 1 fdi_map directly
#   BW: quadrant from image position metadata; tooth from bbox x-position
#   PA: from router metadata or 'unknown_{idx}'
#   CBCT: from nnU-Net centroid mapping
#
# assign_surface(det, tooth_bbox):
#   x_rel = (det.bbox_center_x - tooth_bbox.x1) / tooth_bbox.width
#   y_rel = (det.bbox_center_y - tooth_bbox.y1) / tooth_bbox.height
#   if x_rel < 0.3: mesial
#   elif x_rel > 0.7: distal
#   elif y_rel < 0.25: occlusal
#   elif y_rel > 0.75: cervical
#   else: body
```

### `pipeline/overlay_renderer.py`
```python
# COLOR_MAP = {
#   'enamel_caries':        (80,  80,  255),  # BGR: red
#   'dentin_caries':        (40,  40,  220),
#   'secondary_caries':     (120, 120, 255),
#   'bone_loss':            (30,  160, 255),  # orange
#   'furcation_defect':     (50,  200, 255),
#   'periapical_lesion':    (50,  230, 255),  # yellow
#   'impaction':            (240, 90,  160),  # purple
#   'calculus':             (255, 160, 100),  # blue
#   'existing_restoration': (120, 120, 120),  # gray
# }
#
# SEVERITY_COLORS = {
#   'CRITICAL': (0, 0, 255),   # red
#   'HIGH':     (0, 128, 255), # orange
#   'MEDIUM':   (0, 255, 255), # yellow
#   'LOW':      (0, 255, 0),   # green
# }
#
# render_2d(original_bgr, clusters) -> annotated_bgr:
#   For each cluster:
#     For each finding in cluster.findings:
#       Draw bbox with COLOR_MAP[finding.class_name], thickness=2
#       If mask_rle: decode + draw filled mask at 30% alpha
#     Draw cluster label: "{fdi} | {severity}" above first bbox
#     Draw severity dot: filled circle SEVERITY_COLORS[severity]
#     If flagged_for_review: add dashed border + "? REVIEW" text
#
# render_cbct_3d(seg_path: str) -> rgb_image: np.ndarray:
#   Load segmentation with SimpleITK
#   For each label 1-5: run VTK marching cubes
#   Render with VTK offscreen renderer (vtkRenderWindow.OffScreenRenderingOn)
#   Return RGB numpy array
#
# export(result: InferenceResult, clusters, prompts, images, output_dir):
#   Save: original.png, annotated.png, findings.json, treatment_card.txt, summary.txt
#   CBCT also: mesh_3d.png
#   treatment_card.txt = "\n\n".join(prompt.to_text() for prompt in prompts)
#   summary.txt = result one-paragraph summary
```

### `pipeline/inference_engine.py`
```python
# InferenceEngine — orchestrates all 12 stages
#
# SEQUENCE_MODALITIES = {ModalityType.CBCT, ModalityType.FMS, ModalityType.SERIAL_BW}
#
# __init__(config, checkpoint_dir, device):
#   Load all models to device
#   Load templates
#   Log model versions to audit trail
#
# infer(input_path, modality, patient_context=None, use_vlm=False) -> InferenceResult:
#   t0 = time.perf_counter()
#   try:
#     images, meta = loader.load(input_path)
#     quality_gate.check_all(images)                          # Stage 1
#     modality = router.classify(input_path, meta)            # Stage 2
#     preprocessed = preprocessor.process(images, modality)  # Stage 3
#     features = dentvfm.encode(preprocessed)                # Stage 4
#     if modality in SEQUENCE_MODALITIES:                    # Stage 5
#       features = c1.forward(features, meta.slice_spacing)
#     raw_dets = task_heads[modality].detect(features)        # Stage 6
#     if modality == CBCT:
#       seg = nnunet.segment(input_path)                     # Stage 6 parallel
#     if modality in SEQUENCE_MODALITIES:                    # Stage 7
#       raw_dets = c2.filter(raw_dets, meta.slice_spacing)
#     dets = postprocessor.process(raw_dets, modality)        # Stage 8
#     clusters = c3.cluster(dets)                             # Stage 9
#     if modality == CBCT:
#       clusters = c3.merge_with_3d(clusters, seg)
#     annotated = renderer.render(images, clusters)           # Stage 10
#     prompts = c4.generate(clusters, patient_context,        # Stage 11
#                            use_vlm=use_vlm)
#     result = InferenceResult(...)                          # Stage 12
#   except QualityError as e:
#     result = InferenceResult(error=str(e), ...)
#   result.processing_time_ms = (time.perf_counter() - t0) * 1000
#   return result
```

---

## `models/medclip/` — Med-CLIP IMPLEMENTATIONS

### `models/medclip/biomedclip.py`
```python
# BiomedCLIPEncoder — PRIMARY production Med-CLIP (use this everywhere)
#
# Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# Architecture: ViT-B/16 image encoder + PubMedBERT text encoder
# Trained on: PMC-15M (15M biomedical figure-caption pairs from PubMed Central)
# HuggingFace: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
#
# Load:
#   from open_clip import create_model_from_pretrained, get_tokenizer
#   model, preprocess = create_model_from_pretrained(
#       'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
#   )
#   tokenizer = get_tokenizer(
#       'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
#   )
#
# class BiomedCLIPEncoder:
#   __init__(device='cuda', freeze=True):
#     self.model = model (from above)
#     self.preprocess = preprocess  # standard 224×224 normalization
#     self.tokenizer = tokenizer
#     if freeze: model.requires_grad_(False)
#     self.image_embed_dim = 512
#     self.text_embed_dim = 512
#
#   encode_image(image_tensor: Tensor[B,3,224,224]) -> Tensor[B, 512]
#     return F.normalize(model.encode_image(image_tensor), dim=-1)
#
#   encode_text(texts: List[str]) -> Tensor[N, 512]
#     tokens = tokenizer(texts, context_length=256)
#     return F.normalize(model.encode_text(tokens), dim=-1)
#
#   zero_shot_classify(image_tensor, class_texts, template="{}") -> Tensor[B, N]
#     text_feats = encode_text([template.format(t) for t in class_texts])
#     img_feats = encode_image(image_tensor)
#     return (img_feats @ text_feats.T) * model.logit_scale.exp()
#
#   dental_zero_shot(image_tensor, threshold=0.35) -> List[str]
#     DENTAL_PROMPTS = {
#       "enamel_caries":    "dental caries in enamel on bitewing radiograph",
#       "dentin_caries":    "dentin caries on dental X-ray radiograph",
#       "deep_caries":      "deep caries near pulp on dental radiograph",
#       "periapical_lesion":"periapical lesion on dental X-ray endodontic",
#       "bone_loss":        "alveolar bone loss periodontal on dental X-ray",
#       "impaction":        "impacted tooth on panoramic dental radiograph",
#       "calculus":         "dental calculus tartar on radiograph",
#       "healthy":          "healthy normal tooth dental radiograph",
#     }
#     scores = zero_shot_classify(image_tensor, list(DENTAL_PROMPTS.values()))
#     scores = scores.softmax(dim=-1)
#     return [k for k, s in zip(DENTAL_PROMPTS, scores[0]) if s > threshold
#             and k != 'healthy']
#
# USAGE in pipeline:
#   router.py:     dental_zero_shot for modality classification
#   panoramic_head.py: zero_shot_classify per tooth ROI (Stage 2)
#   c4_prompts.py: encode_image for visual grounding in Tier 2
```

### `models/medclip/mmkd_clip.py`
```python
# MMKDCLIPEncoder — UPGRADE PATH (use when publicly released)
# Paper: "Unifying Biomedical Vision-Language Expertise via Multi-CLIP
#          Knowledge Distillation" arXiv 2506.22567
#
# What it is: Knowledge distillation from 9 CLIP models:
#   BMCA, GenMedClip, MedCLIP, BiomedCLIP, UniMedCLIP,
#   PLIP, PubMedCLIP, QuiltNet, PMC-CLIP
# Performance: #1 AUC on X-ray in 7/9 imaging modalities (2025)
# HuggingFace: TBD — check paper for release
#
# class MMKDCLIPEncoder:
#   __init__(weights_path: str, device='cuda'):
#     # TODO: implement when weights are released
#     # Interface must match BiomedCLIPEncoder exactly
#     raise NotImplementedError(
#         "MMKD-CLIP weights not yet released. "
#         "Using BiomedCLIP. Check arXiv 2506.22567 for release date."
#     )
#
#   encode_image(image_tensor) -> Tensor[B, 512]
#   encode_text(texts) -> Tensor[N, 512]
#   zero_shot_classify(image_tensor, class_texts) -> Tensor[B, N]
#   dental_zero_shot(image_tensor, threshold=0.35) -> List[str]
```

### `models/medclip/regionmed_clip.py`
```python
# RegionMedCLIPEncoder — DENTAL ROI SPECIALIST (use for C4 grounding)
# Paper: "RegionMed-CLIP: Region-Aware Multimodal Contrastive Learning"
#         arXiv 2508.05244
#
# What it is: Region-aware CLIP with explicit DENTAL category
#   Dataset: MedRegion-500k — 12 imaging categories including Dental
#   Each image has: global view + ROI crops + 4 text description types
#   Auto-annotated with Med-SAM + Grounding DINO + Qwen-2.5VL-72B
#
# class RegionMedCLIPEncoder:
#   __init__(weights_path: str, device='cuda'):
#     # TODO: implement when weights released (arXiv 2508.05244)
#     raise NotImplementedError(
#         "RegionMed-CLIP weights not yet released. "
#         "Using BiomedCLIP for grounding. "
#         "Check arXiv 2508.05244 for release."
#     )
#
#   encode_image_region(image, bbox_xyxy) -> Tensor[512]
#     # Key difference from BiomedCLIP: encodes a REGION, not whole image
#     # Crop + context-aware encoding
#   encode_text(texts) -> Tensor[N, 512]
#   dental_region_classify(image, bbox_list) -> List[str]
#     # Use for per-tooth ROI classification in OPG Stage 2
#     # More precise than BiomedCLIP for dental-specific regions
```

### `models/medclip/medclip_factory.py`
```python
# MedCLIPFactory — returns best available model automatically
#
# PRIORITY = ['mmkd_clip', 'biomedclip', 'regionmed_clip']
#
# def get_best_medclip(config, device='cuda'):
#   """Return best available Med-CLIP encoder for this environment."""
#   if config.medclip.mmkd_weights and Path(config.medclip.mmkd_weights).exists():
#       from models.medclip.mmkd_clip import MMKDCLIPEncoder
#       return MMKDCLIPEncoder(config.medclip.mmkd_weights, device)
#   else:
#       from models.medclip.biomedclip import BiomedCLIPEncoder
#       console.print("[yellow]MMKD-CLIP unavailable → using BiomedCLIP[/yellow]")
#       return BiomedCLIPEncoder(device, freeze=True)
#
# def get_grounding_medclip(config, device='cuda'):
#   """Return best Med-CLIP for C4 visual grounding."""
#   if config.medclip.regionmed_weights and Path(config.medclip.regionmed_weights).exists():
#       from models.medclip.regionmed_clip import RegionMedCLIPEncoder
#       return RegionMedCLIPEncoder(config.medclip.regionmed_weights, device)
#   else:
#       from models.medclip.biomedclip import BiomedCLIPEncoder
#       console.print("[yellow]RegionMed-CLIP unavailable → using BiomedCLIP for grounding[/yellow]")
#       return BiomedCLIPEncoder(device, freeze=True)
```

---

## TRAINING PHASES

### `training/phase1_pretrain.py`
```python
# DINOv2 self-supervised pre-training on ALL dental images (no labels)
# Data: BW + OPG + PA + CBCT slices extracted to 2D
# Use: facebookresearch/dinov2 loss implementation (install from GitHub)
# Model: ViT-L/14 (timm initialization)
# Optimizer: AdamW, lr=1e-4, cosine decay, warmup=10epochs
# Augmentation: RandomResizedCrop(518), ColorJitter(0.4,0.4,0.2,0.1),
#               GaussianBlur(p=0.5), RandomHorizontalFlip
# Batch: 256 (gradient accumulation over 8 steps if needed)
# Log: loss, ssl_metrics to wandb project "dentalmine-phase1"
# Save checkpoint every 10 epochs + best
# --fast-dev-run: 1 batch only
```

### `training/phase2_c1_adapter.py`
```python
# Train C1 cross-slice attention adapter (DentVFM frozen)
# Data: CBCT volumes → extract slice sequences with 3D labels
# Supervision: project nnU-Net 3D labels to per-slice binary maps
# Loss: BCEWithLogitsLoss for per-slice detection targets
# Optimizer: AdamW, lr=2e-4, cosine schedule
# K curriculum: epochs 1-10 K=1, epochs 11-30 K=2, epochs 31+ K=3
# Validation metric: per-slice detection mAP + 3D consistency score
# (3D consistency = fraction of detections surviving C2 filter)
```

### `training/phase3_multitask.py`
```python
# Simultaneous fine-tuning of all task heads (DentVFM frozen)
# JointDataset samples from all modalities with weights:
#   BW=0.35, OPG=0.35, PA=0.15, CBCT_2.5D=0.15
# Total loss: sum of per-modality losses (balanced by gradient scaling)
# Loss per head:
#   BW:   ultralytics YOLOv8 detection + segmentation loss
#   OPG:  Stage1 YOLO loss + Stage2 BiomedCLIP zero-shot contrastive loss
#         (BiomedCLIP features as soft targets for Stage 2 head fine-tuning)
#   PA:   YOLOv8 detection loss
#   CBCT: BCE detection loss on 2.5D outputs
# AUXILIARY LOSS (add weight=0.1):
#   BiomedCLIP alignment loss: encourage DentVFM features to align with
#   BiomedCLIP embeddings for same dental images (knowledge distillation
#   from BiomedCLIP into DentVFM). MSE(dentvfm_proj(feat), biomedclip_feat.detach())
#   where dentvfm_proj is a Linear(1024→512) projection layer
# Optimizer: AdamW lr=1e-4, cosine schedule, weight_decay=0.01
# EarlyStopping: patience=15 epochs on val mAP (combined)
# DRR augmentation: add DRRs from cbct to OPG + PA batches
```

### `training/phase4_distillation.py`
```python
# 3D → 2D knowledge distillation loop
# Step 1: Run nnU-Net on all CBCT in cbct_dir → 3D masks
# Step 2: For each CBCT: generate DRR panoramic + 14 PA projections
#         using: subprocess.run(['plastimatch', 'drr', ...])
# Step 3: Project 3D tooth masks onto DRR planes → 2D pseudo-labels
#         Keep only predictions where nnU-Net confidence (softmax max) > 0.7
# Step 4: Write pseudo-labeled DRRs to pseudo_output/opg/ and pseudo_output/pa/
# Step 5: Print instructions to re-run Phase 3 with --include-pseudo flag
# Repeat up to 3 iterations (implemented as 3 sequential script runs)
```

---

## CONFIG FILES

### `config/default.yaml`
```yaml
model:
  encoder: vit_large_patch14_dinov2
  encoder_checkpoint: null
  dentvfm_weights: null
  dentalsegemtator_weights: null
  c1_k_cbct: 3
  c1_k_fms: 1
  c1_k_serial_bw: 2
  use_vlm: false
  vlm_model: Qwen/Qwen2-VL-7B-Instruct

medclip:
  # Primary Med-CLIP — production ready, auto-downloaded from HuggingFace
  biomedclip_model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
  biomedclip_context_length: 256
  biomedclip_threshold: 0.35       # zero-shot classification threshold
  biomedclip_template: "{}"        # prompt template; "{}" = class name as-is

  # Upgrade path — set path when MMKD-CLIP weights released (arXiv 2506.22567)
  # Med-CLIP model selection
  # PRIMARY: MMKD-CLIP (ViT-B-16-quickgelu, distills 9 biomedical CLIPs)
  # GitHub: github.com/wangshansong1/MMKD-CLIP
  # Weights: Google Drive 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR
  mmkd_weights: ./weights/MMKD_B16.pth   # set to null to force BiomedCLIP fallback
  mmkd_model_name: ViT-B-16-quickgelu
  mmkd_text_encoder: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
  mmkd_context_length: 256

  # FALLBACK: BiomedCLIP (auto-downloaded from HuggingFace if MMKD absent)
  biomedclip_hf_id: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

  # Dental ROI grounding — set path when RegionMed-CLIP released (arXiv 2508.05244)
  regionmed_weights: null          # null = use MMKD-CLIP for grounding

  # MMKD-CLIP→DentVFM distillation (Phase 3 auxiliary loss)
  alignment_loss_weight: 0.10      # weight of MMKD-CLIP alignment auxiliary loss
  alignment_projection_dim: 512    # Linear(1024→512) to match MMKD-CLIP D=512

inference:
  nms_iou_threshold: 0.45
  detection_score_threshold: 0.35
  calibration_temperature: 1.3
  review_flag_threshold: 0.55
  c2_min_support: 3
  c2_consistency_iou: 0.40
  c2_physical_window_mm: 1.0

training:
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  max_epochs: 100
  warmup_epochs: 5
  modality_weights:
    bitewing: 0.35
    panoramic: 0.35
    periapical: 0.15
    cbct_25d: 0.15

data:
  dentex_path: ./data/dentex/
  bitewing_path: ./data/bitewing/
  cbct_path: ./data/cbct/
  drr_output_path: ./data/drr/
  pseudo_label_path: ./data/pseudo/
  yolo_image_size: 640
  dentvfm_image_size: 518

quality:
  min_snr_db: 15.0
  min_entropy_bits: 4.5
  max_overexposed_fraction: 0.40

rendering:
  overlay_alpha: 0.30
  show_confidence: true
  show_fdi_labels: true
  cbct_3d_render: true
  output_formats: [png, json, txt]
```

### `templates/treatment_templates.yaml`
```yaml
# Implement all 11 patterns. Example:
caries_dentin:
  description: "Dentin caries detected on {surface} surface of tooth {tooth}"
  options:
    - rank: 1
      text: "Composite resin restoration after caries removal"
      when_to_use: "Vital tooth, no symptoms, caries not near pulp"
    - rank: 2
      text: "Pulp vitality test before restoration; consider indirect pulp cap"
      when_to_use: "Caries radiographically near pulp"
    - rank: 3
      text: "Root canal treatment + full coverage crown"
      when_to_use: "Irreversible pulpitis or caries into pulp"
  tests_suggested:
    - "Cold vitality test"
    - "Percussion test"
    - "Periapical radiograph for apical status"
  red_flags:
    - "Spontaneous or lingering pain → urgent RCT referral"
    - "Swelling or sinus tract → immediate treatment"
    - "Radiographic periapical lesion alongside caries → endo complex pattern"

# Add remaining 10 patterns:
# caries_enamel, caries_pulp_risk, caries_with_bone, caries_with_pal,
# bone_loss_only, multi_tooth_bone, impaction, endo_complex,
# calculus_only, restoration_issue
```

---

## `scripts/download_weights.py`
```python
# Download all pretrained weights needed.
# Run: python scripts/download_weights.py --output ./weights/
#
# 1. MMKD-CLIP (PRIMARY Med-CLIP — download first):
#    Weights: Google Drive file ID 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR
#    Command: gdown 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR -O ./weights/MMKD_B16.pth
#    Size: ~400 MB
#    Also clone source repo (needed for get_mean_std and HFTokenizer):
#      git clone https://github.com/wangshansong1/MMKD-CLIP external/MMKD-CLIP
#      pip install -r external/MMKD-CLIP/requirements.txt
#    After download, verify with:
#      python -c "
#      import sys; sys.path.insert(0,'external/MMKD-CLIP/src')
#      from models.medclip.mmkd_clip import MMKDCLIPEncoder
#      enc = MMKDCLIPEncoder('./weights/MMKD_B16.pth', device='cpu')
#      import torch
#      out = enc.dental_zero_shot(torch.zeros(1,3,224,224))
#      print('MMKD-CLIP OK:', list(out.keys())[:3])
#      "
#
# 2. DentalSegmentator (nnU-Net CBCT):
#    URL: https://zenodo.org/records/10829675
#    Save to: weights/dental_segmentator/
#
# 3. DentVFM-2D (if publicly released by paper authors):
#    Check: https://github.com/dentvfm/DentVFM (or arxiv link)
#    Fallback message: "DentVFM weights not yet public. Using MMKD-CLIP ViT-B/16."
#
# 4. HierarchicalDet:
#    git clone https://github.com/ibrahimethemhamamci/HierarchicalDet external/HierarchicalDet
#    pip install -e external/HierarchicalDet/
#
# 5. BiomedCLIP (FALLBACK — auto-downloads on first use if MMKD absent):
#    huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
#    Only needed if ./weights/MMKD_B16.pth is missing
#
# 6. RegionMed-CLIP (when released — check arXiv 2508.05244):
#    Print: "RegionMed-CLIP not yet released. Using MMKD-CLIP for grounding."
#
# Print a summary table at end (use rich.table):
#   ✓ MMKD-CLIP           — ready at weights/MMKD_B16.pth  (PRIMARY)
#   ✓ DentalSegmentator   — ready at weights/dental_segmentator/
#   ? DentVFM-2D          — not available; using MMKD-CLIP ViT-B/16 as fallback
#   ? BiomedCLIP          — fallback only; MMKD-CLIP is active
#   ? RegionMed-CLIP      — not yet released; using MMKD-CLIP for grounding
#   ✓ HierarchicalDet     — cloned to external/HierarchicalDet/
```

---

## OUTPUT REQUIREMENTS

Every `dentalmine infer` run must produce:
```
{output_dir}/
├── findings.json          # valid InferenceResult JSON
├── original.png           # original image
├── annotated.png          # color overlay
├── treatment_card.txt     # per-tooth prompts + disclaimer
├── summary.txt            # one paragraph: N teeth affected, highest severity, top recommendation
└── cbct_3d.png            # (CBCT only)
```

`findings.json` validates against `InferenceResult` Pydantic schema.
Every `treatment_card.txt` page ends with the disclaimer string.
Audit log must record: model versions, inference time, flagged detections count.
Always anonymize: strip PatientID, PatientName, DOB from any DICOM output.

---

## TESTS (`tests/`)

```python
# test_schemas.py
# - InferenceResult validates and serializes to JSON
# - PromptOutput always has disclaimer
# - Detection bbox values in [0,1]

# test_madclip.py
# - C1: dummy tensors [2,37,37,1024] → output same shape
# - C1: single image returns identity (no error)
# - C2: inject 10 detections, 8 noise (0 neighbors), 2 real (4 neighbors)
#       → exactly 2 survive at min_support=3
# - C3: 6 detections across 3 FDI teeth → 3 clusters
# - C4: all 11 patterns produce valid PromptOutput with disclaimer

# test_pipeline_e2e.py
# - Load fixture OPG JPEG → full infer() → valid InferenceResult
# - Output files exist in output_dir
# - processing_time_ms > 0
# All tests run CPU-only (no GPU required)
```

```bash
# All tests must pass:
pytest tests/ -v --tb=short
```

---

## IMPLEMENTATION ORDER

Implement in this exact sequence:

1. `pyproject.toml` + `requirements.txt` + project skeleton
2. All schemas (`schema/`)
3. `config/default.yaml` + `templates/treatment_templates.yaml` (all 11 patterns)
4. `data/quality/quality_gate.py`
5. `external/MMKD-CLIP/` clone + `pip install -r requirements.txt`
6. `models/medclip/mmkd_clip.py` ← FIRST: implement MMKDCLIPEncoder with exact loading code from README
7. `models/medclip/biomedclip.py` ← SECOND: implement as fallback with same API
8. `models/medclip/medclip_factory.py` ← factory: try MMKD_B16.pth → BiomedCLIP → error
9. `pipeline/router.py` (use MedCLIPFactory zero-shot for modality classification)
10. `pipeline/preprocessor.py`
11. `models/backbone/dentvfm.py` (MMKD-CLIP as fallback encoder via factory)
12. `models/madclip/c3_clustering.py` (Strategy 1 only first)
13. `models/madclip/c4_prompts.py` (Tier 1 templates + MMKD-CLIP grounding stub)
14. `models/heads/panoramic_head.py` (Stage 1 YOLOv11 + Stage 2 MMKD-CLIP zero-shot)
15. `pipeline/postprocessor.py`
16. `pipeline/overlay_renderer.py` (2D only first)
17. `pipeline/inference_engine.py`
18. `cli.py` (infer command only first)
19. `tests/test_schemas.py` + `tests/test_madclip.py` + `tests/test_pipeline_e2e.py`
20. `models/madclip/c1_slice_attention.py`
21. `models/madclip/c2_consistency.py`
22. `models/heads/bitewing_head.py`
23. `models/heads/periapical_head.py`
24. `models/heads/cbct_head.py`
25. `models/segmentation/nnunet_wrapper.py`
26. `models/vlm/dentvlm.py`
27. `models/medclip/regionmed_clip.py` (stub with NotImplementedError + TODO)
28. `pipeline/overlay_renderer.py` (add CBCT 3D rendering)
29. All training phases (`training/`) — include MMKD-CLIP alignment loss in phase3
30. All scripts (`scripts/`) — `download_weights.py` downloads MMKD_B16.pth via gdown first
31. Remaining CLI commands (train, eval, serve, data)
32. `README.md` with setup + quickstart

---

## FINAL VALIDATION

After implementing everything, run:
```bash
pip install -e .
pytest tests/ -v

# Step 1 — install MMKD-CLIP and download weights (run once)
git clone https://github.com/wangshansong1/MMKD-CLIP external/MMKD-CLIP
pip install -r external/MMKD-CLIP/requirements.txt
gdown 16pDQ2rI3VKezUl_WVxm1gVmhljQWxjJR -O weights/MMKD_B16.pth

# Step 2 — verify MMKD-CLIP loads correctly (most critical dependency)
python -c "
import sys; sys.path.insert(0,'external/MMKD-CLIP/src')
from models.medclip.mmkd_clip import MMKDCLIPEncoder
import torch
enc = MMKDCLIPEncoder('./weights/MMKD_B16.pth', device='cpu')
scores = enc.dental_zero_shot(torch.zeros(1,3,224,224))
assert len(scores) == 7, 'Expected 7 dental prompts'
print('MMKD-CLIP OK — top finding:', max(scores, key=scores.get))
"

# Step 3 — verify factory fallback
python -c "
from models.medclip.medclip_factory import MedCLIPFactory
enc = MedCLIPFactory.get_encoder(mmkd_path='weights/MMKD_B16.pth', device='cpu')
print('Factory loaded:', type(enc).__name__)
"

# Step 4 — download data and smoke test pipeline
dentalmine data download-weights --output ./weights/
dentalmine data download-dentex --output ./data/dentex/
dentalmine train phase3 --config config/default.yaml --fast-dev-run
dentalmine infer --input tests/fixtures/test_opg.jpg --modality opg --output ./smoke_test/
cat ./smoke_test/treatment_card.txt
cat ./smoke_test/findings.json | python -c "import sys,json; json.load(sys.stdin); print('JSON valid')"
```

All commands must succeed without error.
