# DentalMind — package guide

Dental X-ray AI pipeline: detect pathology, cluster findings per tooth (FDI),
and generate ranked treatment prompts — across panoramic (OPG), bitewing,
periapical and (when data is available) CBCT.

This document explains **the shared CLIP baseline first** (the core of the
project), then the inference pipeline, then every other runnable part.

---

## ⭐ 1. THE SHARED CLIP BASELINE (the most important part)

### What it is
A **single Med-CLIP image encoder is the shared backbone for every task, in both
2D and 3D.** There is no separate per-modality backbone — the same weights encode
a 2D tooth crop and a 3D CBCT slice. This is the project's central design bet:
one strong, dentally-adapted vision-language backbone, with lightweight task- and
dimensionality-specific pieces bolted on top.

```
                    ┌─────────────────────────────┐
                    │   Shared Med-CLIP encoder    │   MMKD-CLIP (primary)
                    │   (image tower, fine-tuned)  │   → BiomedCLIP (fallback)
                    └──────────────┬──────────────┘
              2D crops             │            CBCT slices
        ┌─────────────────────────┴──────────────────────────┐
        ▼                                                      ▼
  2D BRANCH                                              3D BRANCH
  prototype fine-tune on FDI                       same encoder embeds each
  tooth + caries/healthy crops                     slice → C1 cross-slice
  (text-prototype CE loss)                         attention fuses a window
        │                                          of neighbours → slice head
        └──────────────► one shared backbone ◄─────────────────┘
```

The 3D half does **not** introduce a new backbone: it reuses the identical
(now dentally-adapted) CLIP encoder per slice and only adds the small **C1
cross-slice attention** module to give the 2D backbone 3D awareness. That is what
"the CLIP baseline combines 2D and 3D" means here — literally shared weights.

### Where it lives
- `training/baseline_clip_2d3d.py` — the unified entry point (2D branch + 3D branch)
- `training/clip_finetune.py` — the 2D fine-tune loop (`finetune_2d`), reused by the baseline
- `models/madclip/c1_slice_attention.py` — the C1 cross-slice module used by the 3D branch
- `models/medclip/medclip_factory.py` — picks MMKD-CLIP if weights exist, else BiomedCLIP

### Completeness status (honest)
| Part | State |
|---|---|
| Shared encoder selection (MMKD-CLIP → BiomedCLIP) | ✅ complete |
| 2D branch — fine-tune on FDI + caries crops, leak-free held-out val | ✅ complete, runs on your data (verified: 24,384 train / 2,690 val crops, no image overlap) |
| 3D branch — same encoder + C1 cross-slice attention | ✅ code complete & smoke-verified; **runs for real only once CBCT data exists** |
| CBCT data | ❌ not downloaded yet (`data/3D` empty) — 3D branch auto-skips with a clear message |

So: the **2D+CLIP baseline is complete and trainable today**; the **3D half is
implemented and shares the same CLIP backbone**, and activates the moment you
drop CBCT volumes + masks in place. Nothing is faked — with no 3D data the 3D
branch prints exactly what's missing and stops.

### How to run it
```bash
cd dentalmine

# quick end-to-end code-path check (seconds; 2D does 1 batch, 3D synthetic):
python -m training.baseline_clip_2d3d --data-root ../data/2D --fast-dev-run

# real run on the A100 (2D for real; 3D auto-runs if CBCT present, else skips):
python -m training.baseline_clip_2d3d --data-root ../data/2D --device cuda \
    --epochs 10 --batch-size 64

# once you have CBCT volumes + nnU-Net/DentalSegmentator masks, the 3D branch
# trains on the SAME encoder automatically:
python -m training.baseline_clip_2d3d --data-root ../data/2D --device cuda \
    --cbct-dir ../data/3D/cbct --labels-dir ../data/3D/masks
```
Or via the CLI: `dentalmine train baseline --data-root ../data/2D --device cuda`

Outputs: `weights/baseline_clip_2d.pt` (2D-adapted encoder) and, when 3D runs,
`weights/baseline_clip_c1_3d.pt` (the C1 adapter).

> Which encoder is "CLIP"? If `weights/MMKD_B16.pth` exists and `external/MMKD-CLIP`
> is cloned, the baseline uses **MMKD-CLIP**; otherwise it auto-falls back to
> **BiomedCLIP** (auto-downloaded). Set `mmkd_weights` in `config/default.yaml`.

---

## 2. Inference pipeline (Phase 1, OPG end-to-end)

```
image → quality gate → modality router → preprocess → panoramic head
      → postprocess (NMS / FDI / surface / calibration)
      → C3 per-tooth clustering → C4 ranked treatment prompts
      → annotated.png + findings.json + treatment_card.txt + summary.txt
```

```bash
cd dentalmine
pip install -e .

# offline/deterministic (no model downloads):
DENTALMIND_MEDCLIP=stub dentalmine infer \
  --input tests/fixtures/test_opg.jpg --modality opg --output ./results/ --device cpu

# real inference (BiomedCLIP/MMKD encoder):
dentalmine infer --input <your_opg.jpg> --modality opg --output ./results/
```
Outputs in `--output/`: `findings.json` (schema-validated), `original.png`,
`annotated.png` (colour overlay), `treatment_card.txt` (ranked per-tooth options
+ disclaimer), `summary.txt` (plain-language patient summary).

### 2b. CBCT (3D) inference — "after CLIP, see neighbours and decide"
The 3D path is wired into the same engine. Point `--input` at a CBCT volume
(`.nii`/`.nii.gz`) or a DICOM-series directory (auto-detected; or `--modality cbct`):

```bash
dentalmine infer --input scan.nii.gz --modality cbct --output ./results_cbct/ --device cuda
```
Flow (`pipeline/inference_engine.py::_infer_cbct`):
```
volume → axial slices → shared CLIP encodes each slice → per-slice detections
   → C1  SEE NEIGHBOURS : cross-slice attention fuses adjacent slices (trained path)
   → C2  DECIDE         : keep a finding only if it persists across ≥K neighbour
                          slices — a real lesion appears on several slices, single-
                          slice noise is rejected   ← the core 3D reasoning
   → C3 cluster across slices → C4 ranked prompts → representative-slice overlay
```
Verified end-to-end on a synthetic volume: a planted periapical lesion spanning
4 slices survives C2 and becomes an `endo_complex / CRITICAL` finding, while
single-slice noise is removed. Needs `SimpleITK` (in requirements) to read volumes.
The trained 2.5D head that consumes C1-enriched features comes from Phase 2 /
the baseline 3D branch once CBCT training data exists; the current detector is
CLIP zero-shot per slice.

---

## 3. Training phases (PROMPT.md Phases 1–4)

All wired under `dentalmine train …` and runnable as modules. Phases that need
CBCT/3D data are **data-gated** with a `--fast-dev-run` synthetic path that
validates the training code without data.

| Phase | Command | Needs | Runs now? |
|---|---|---|---|
| 1 — SSL pretrain DentVFM backbone | `python -m training.phase1_pretrain --data-dir ../data/2D` | 2D images | ✅ yes |
| 2 — C1 cross-slice adapter | `python -m training.phase2_c1_adapter --cbct-dir … --labels-dir …` | CBCT + masks | ⚠ gated (synthetic `--fast-dev-run`) |
| 3 — multitask heads + CLIP align | `python -m training.phase3_multitask --data-root ../data/2D` | 2D images + Med-CLIP | ✅ yes |
| 4 — 3D→2D distillation | `python -m training.phase4_distillation --cbct-dir … --masks-dir …` | CBCT (+nnU-Net/plastimatch optional) | ⚠ gated (synthetic `--fast-dev-run`) |

Quick validation of all four code paths (seconds each):
```bash
python -m training.phase1_pretrain   --data-dir ../data/2D --fast-dev-run --model vit_base_patch16_224 --img-size 224
python -m training.phase2_c1_adapter --fast-dev-run
python -m training.phase3_multitask  --data-root ../data/2D --fast-dev-run --model vit_base_patch16_224 --img-size 224
python -m training.phase4_distillation --fast-dev-run
```

Real Phase 1 → Phase 3 chain on the A100 (DentVFM backbone, then heads on top):
```bash
python -m training.phase1_pretrain --data-dir ../data/2D --device cuda
python -m training.phase3_multitask --data-root ../data/2D \
    --backbone-ckpt ../weights/dentvfm_phase1.pt --device cuda
```

### How the phases relate to the shared CLIP baseline
- **Baseline (§1)** = the CLIP encoder itself, shared across 2D+3D. This is the
  recommended path for training the shared backbone.
- **Phase 1** trains an *alternative* from-scratch DINOv2-style backbone (DentVFM
  ViT); **Phase 3** then distills CLIP *into* DentVFM via the alignment loss. Use
  these if you specifically want the DentVFM route from PROMPT.md; use the
  **baseline** if you want the CLIP encoder to *be* the shared backbone.

---

## 4. Stage-1 tooth detector (YOLO, real boxes)

Replaces the placeholder FDI grid in the OPG head with a trained detector:
```bash
# convert Tufts tooth boxes → YOLO (verified CCW transform):
python scripts/tufts_to_yolo.py \
  --bbox-json ../data/2D/tufts_raw/Segmentation/teeth_bbox.json \
  --radiographs-dir ../data/2D/tufts_raw/Radiographs --out-dir ../data/2D/tufts_yolo
# train (968 images, 32 FDI classes):
yolo detect train model=yolo11m.pt data=../data/2D/tufts_yolo/data.yaml \
  epochs=150 imgsz=1024 batch=16 device=0
```
Then set `inference.panoramic_stage1_weights` in `config/default.yaml` to the
resulting `best.pt`.

---

## 5. Install & test

```bash
cd dentalmine
pip install -e .            # installs deps from requirements.txt
DENTALMIND_MEDCLIP=stub pytest tests/ -v   # 11 tests, CPU-only
```

> **Always use a dedicated conda/venv for this project.** Installing into a shared
> environment risks numpy/torch/transformers ABI cascades. `transformers` must be
> `<5.0` (5.x breaks MMKD-CLIP's vendored open_clip fork and BiomedCLIP loading).

---

## 6. Module map

```
dentalmine/
├── cli.py                       infer / infer-batch / train {baseline,phase1..4,clip-backbone}
├── config/default.yaml          model + inference + data config
├── schema/                      finding / prompt / patient pydantic schemas
├── data/
│   ├── loaders/image_loader.py  2D / DICOM loading
│   └── quality/quality_gate.py  SNR (Immerkär) / entropy / exposure gate
├── models/
│   ├── backbone/dentvfm.py      shared ViT encoder (Phase 1/3)
│   ├── medclip/                 MMKD-CLIP → BiomedCLIP → Stub factory
│   ├── madclip/                 c1 (cross-slice), c2 (consistency), c3 (cluster), c4 (prompts)
│   └── heads/panoramic_head.py  two-stage OPG head (YOLO + Med-CLIP zero-shot)
├── pipeline/                    router, preprocessor, postprocessor, overlay, inference_engine
├── training/
│   ├── baseline_clip_2d3d.py    ⭐ unified shared-CLIP 2D+3D baseline
│   ├── clip_finetune.py         2D CLIP fine-tune (finetune_2d, reused by baseline)
│   ├── phase1..4_*.py           PROMPT.md training phases
│   └── trainer_base.py          shared loop utilities
└── scripts/tufts_to_yolo.py     Tufts → YOLO FDI dataset converter
```
