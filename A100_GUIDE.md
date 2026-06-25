# DentalMind — A100 Setup, Inference & Training Guide

This guide covers running DentalMind on your **A100 80 GB SSH server**: what is
implemented today, how to set it up, how to run inference, and how to train.

> **Honesty note on training status**
> - ✅ **Inference pipeline (Phase 1, OPG)** — fully implemented and tested.
> - ✅ **Stage-1 tooth detector training on DENTEX via `ultralytics`** — runnable
>   today (standard YOLO training; recipe below).
> - 🚧 **Custom training phases 1–4** (`dentalmine train phaseN`) — **NOT yet
>   implemented**. The CLI subcommands are scaffolded but raise
>   `NotImplementedError`. See [Training roadmap](#training-roadmap).

---

## 1. What is implemented (Phase 1)

End-to-end **panoramic (OPG)** pipeline, with the integration layer (C3/C4) that
turns detections into per-tooth diagnoses and ranked treatment prompts:

```
image → quality gate → modality router → preprocess → panoramic head
      → postprocess (NMS / FDI / surface / calibration)
      → C3 per-tooth clustering → C4 treatment prompts
      → annotated.png + findings.json + treatment_card.txt + summary.txt
```

| Component | Module | State |
|---|---|---|
| Pydantic schemas | `schema/` | ✅ |
| Quality gate (Immerkær SNR, entropy, exposure) | `data/quality/quality_gate.py` | ✅ |
| Med-CLIP: MMKD → BiomedCLIP → Stub | `models/medclip/` | ✅ |
| Modality router | `pipeline/router.py` | ✅ |
| Preprocessor (CLAHE, crop, normalize) | `pipeline/preprocessor.py` | ✅ |
| Panoramic head (YOLO stage-1 + Med-CLIP zero-shot stage-2) | `models/heads/panoramic_head.py` | ✅ |
| C3 per-tooth clustering | `models/madclip/c3_clustering.py` | ✅ |
| C4 ranked treatment prompts (Tier-1 templates) | `models/madclip/c4_prompts.py` | ✅ |
| Postprocessor | `pipeline/postprocessor.py` | ✅ |
| Overlay renderer (2D) | `pipeline/overlay_renderer.py` | ✅ |
| Inference engine | `pipeline/inference_engine.py` | ✅ |
| CLI `infer` / `infer-batch` | `cli.py` | ✅ |
| Tests (CPU-only) | `tests/` | ✅ 11 passing |

**Not yet implemented** (roadmap): bitewing/periapical/CBCT heads, MadClip C1
(cross-slice attention) & C2 (consistency filter), nnU-Net 3D segmentation,
DentVFM backbone, custom training phases, C4 Tier-2 VLM (Qwen2-VL).

---

## 2. A100 environment setup

### 2.1 Copy the project to the server

```bash
# from your laptop (repo root: d:\git\dental\medclip)
rsync -avz --exclude '.venv' --exclude 'datasets/*.zip' --exclude '__pycache__' \
  ./ user@A100_HOST:~/dentalmind/
# or: scp -r dentalmine A100_GUIDE.md DATASETS.md user@A100_HOST:~/dentalmind/
```

### 2.2 Create the environment (on the A100)

```bash
ssh user@A100_HOST
cd ~/dentalmind
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip

# PyTorch for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# DentalMind + Phase-1 runtime deps
cd dentalmine
pip install -e .
pip install open_clip_torch ultralytics pydicom

# verify GPU
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2.3 Get the data onto the A100 (small datasets — no 10 GB download needed)

You do **not** need the 10 GB DENTEX training archive to start training. The
Stage-1 tooth detector (§4) trains on the **Tufts** dataset instead — real
tooth-level bounding boxes, only ~310 MB, already converter-verified (§4.1).

```bash
mkdir -p ~/dentalmind/data && cd ~/dentalmind/data

# --- Tufts Dental Database (~310 MB) — real tooth bboxes, used for training ---
pip install kaggle
# auth: either drop kaggle.json in ~/.kaggle/, or use a token:
export KAGGLE_API_TOKEN=<your_token>     # from kaggle.com/settings -> API
mkdir -p tufts_raw && kaggle datasets download \
  -d tommyngx/the-tufts-dental-database-2022 -p tufts_raw --unzip

# the converter (scripts/tufts_to_yolo.py) expects this layout; the Kaggle zip
# unpacks slightly differently per mirror version — check and adjust paths:
#   tufts_raw/Radiographs/*.JPG               (panoramic images)
#   tufts_raw/Segmentation/teeth_bbox.json    (tooth bounding boxes)

# --- optional bonus: Zenodo Dental Cavity Dataset (~107 MB, single-class
#     "cavity" Pascal-VOC boxes on periapical x-rays — useful later for a
#     periapical/caries detector, not required for Stage-1 FDI training) ---
curl -L -C - -o Dental_Cavity_Dataset.zip \
  "https://zenodo.org/api/records/4907880/files/Dental%20Cavity%20Dataset.zip/content"
unzip -q Dental_Cavity_Dataset.zip -d caries_zenodo
```

If you'd rather skip re-downloading and just copy the **already-converted**
118 MB YOLO dataset built and verified on the laptop:

```bash
# from the laptop:
rsync -avz d:/git/dental/medclip/data/tufts_yolo/ user@A100_HOST:~/dentalmind/data/tufts_yolo/
```

The 10 GB DENTEX training archive remains useful later (quadrant/disease labels,
full DENTEX-scale pretraining) but is **not required** for the training recipe
below — see `DATASETS.md` for that download if/when you want it.

---

## 3. Running inference on the A100

```bash
cd ~/dentalmind/dentalmine

# real Med-CLIP (BiomedCLIP auto-downloads ~430 MB on first run)
dentalmine infer \
  --input ../data/tufts_raw/Radiographs/1.JPG \
  --modality opg --output ./results/ --device cuda

# offline / deterministic sanity check (no model downloads)
DENTALMIND_MEDCLIP=stub dentalmine infer \
  --input tests/fixtures/test_opg.jpg --modality opg --output ./results/ --device cpu
```

Outputs in `--output/`: `findings.json`, `original.png`, `annotated.png`,
`treatment_card.txt`, `summary.txt`.

Run the test suite:

```bash
DENTALMIND_MEDCLIP=stub pytest tests/ -v
```

---

## 4. Training the Stage-1 tooth detector (RUNNABLE TODAY)

This is the highest-value training step for Phase 1: a real YOLOv11 tooth
detector replaces the placeholder FDI grid, giving genuine per-tooth boxes +
FDI numbering. It trains on the **Tufts** dataset (no 10 GB download needed)
and uses `ultralytics` directly — no custom training loop required.

### 4.1 Convert Tufts annotations → YOLO format (already verified)

`scripts/tufts_to_yolo.py` converts `teeth_bbox.json` (Universal Numbering
1–32, Labelbox canvas coordinates) into a standard YOLO dataset with FDI class
names. **The coordinate transform has been verified two ways**: (1) numerically
— same-side tooth pairs (e.g. Universal #1 upper-right-3rd-molar and #32
lower-right-3rd-molar) land in matching x-ranges with correct upper/lower
y-halves, and (2) visually — rendering decoded boxes back onto the source
radiographs shows them landing tightly on individual teeth across the full
dental arch (use `--visualize-n 5` to re-check yourself anytime).

```bash
cd ~/dentalmind/dentalmine
python scripts/tufts_to_yolo.py \
  --bbox-json ../data/tufts_raw/Segmentation/teeth_bbox.json \
  --radiographs-dir ../data/tufts_raw/Radiographs \
  --out-dir ../data/tufts_yolo \
  --val-fraction 0.1 --visualize-n 5
```

Output (skip this step entirely if you rsync'd the pre-converted folder in §2.3):

```
data/tufts_yolo/
├── data.yaml          # 32 FDI classes, train/val paths
├── images/{train,val}/*.jpg   (968 images total)
└── labels/{train,val}/*.txt   (25,184 tooth boxes)
```

Primary teeth (letter titles A–T, pediatric mixed-dentition scans) are skipped —
out of scope for Phase 1's permanent-tooth FDI vocabulary (`11`–`48`).

### 4.2 Train on the A100

```bash
yolo detect train \
  model=yolo11m.pt \
  data=~/dentalmind/data/tufts_yolo/data.yaml \
  epochs=150 imgsz=1024 batch=32 device=0 \
  project=~/dentalmind/runs name=tufts_fdi
```

968 images is a small dataset for 32 classes — expect decent but not
state-of-the-art mAP. Treat this as the first real checkpoint to replace the
placeholder grid; revisit with DENTEX (§2.3 note) once you want more scale.

### 4.3 Plug the checkpoint into the pipeline

Edit `dentalmine/config/default.yaml`:

```yaml
inference:
  panoramic_stage1_weights: ~/dentalmind/runs/tufts_fdi/weights/best.pt
```

Now `dentalmine infer` uses real YOLO tooth detection (`_stage1_yolo`) instead of
the synthetic grid. The class names in the checkpoint are the FDI strings
(`"11".."48"`); the converter sets this up automatically.

---

## 5. Training roadmap (NOT yet implemented)

The commands below exist as **stubs** (`dentalmine train phaseN` →
`NotImplementedError`). They follow PROMPT.md and will be built in later phases.
Listed here so the intended interface is clear:

```bash
# Phase 1 — DINOv2 self-supervised pretrain of DentVFM ViT-L on ALL dental images
dentalmine train phase1 --data-dir ./data/ --epochs 100 --fast-dev-run

# Phase 2 — train C1 cross-slice attention adapter on CBCT (DentVFM frozen)
dentalmine train phase2 --cbct-dir ./data/cbct/ --backbone-ckpt ./ckpt/phase1.pt

# Phase 3 — multitask fine-tune all heads (+ MMKD-CLIP alignment aux loss)
dentalmine train phase3 --config ./config/default.yaml --fast-dev-run

# Phase 4 — 3D→2D distillation (nnU-Net pseudo-labels via DRR)
dentalmine train phase4 --cbct-dir ./data/cbct/ --pseudo-output ./data/pseudo/
```

**To make these real, the following must be implemented** (rough order/effort):
1. `data/datasets/dentex_dataset.py` + `joint_dataset.py` (PyTorch datasets,
   DENTEX/caries loaders, modality sampling weights).
2. `models/backbone/dentvfm.py` (ViT-L/14 wrapper; DINOv2 / MMKD fallback).
3. `training/trainer_base.py` (loop, AMP, wandb, ckpt, `--fast-dev-run`).
4. `training/phase3_multitask.py` first (most useful: trains the detection
   heads), then phase1 pretrain, then phase2/4 (need CBCT data — none downloaded
   yet).

> Phases 2 & 4 require a CBCT dataset, which is **not in the current download
> set** (`data/cbct/` is empty). Phase 3 + the Stage-1 YOLO recipe in §4 are the
> practical near-term training paths.

---

## 6. Quick reference

| Task | Command |
|---|---|
| Install | `pip install -e . && pip install open_clip_torch ultralytics pydicom` |
| GPU check | `python -c "import torch;print(torch.cuda.is_available())"` |
| Inference | `dentalmine infer --input X.png --modality opg --output ./out --device cuda` |
| Tests | `DENTALMIND_MEDCLIP=stub pytest tests/ -v` |
| Train Stage-1 YOLO | `yolo detect train model=yolo11m.pt data=.../data.yaml epochs=150 imgsz=1024 batch=32 device=0` |
| Custom phases | 🚧 not implemented (stubs) |
