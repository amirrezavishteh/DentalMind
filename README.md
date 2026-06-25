# DentalMind

A dental X-ray AI pipeline: detect pathology, cluster findings per tooth, and
generate ranked treatment suggestions for dentists and patients — from
panoramic (OPG), bitewing, periapical, full-mouth-series, and CBCT images.

```
image → quality gate → modality router → preprocess → detection head(s)
      → postprocess (NMS / FDI / surface / calibration)
      → per-tooth clustering (MadClip C3) → treatment prompts (MadClip C4)
      → annotated overlay + findings.json + treatment card + summary
```

The core design bet: **detection accuracy matters less than integration**. A
system that finds 85% of caries but explains exactly what it means per tooth
and what to do beats one that finds 90% but dumps fragmented detections on the
dentist. So Phase 1 builds the full per-tooth clustering + treatment-prompt
pipeline first, on top of even a placeholder detector — see `PROMPT.md` for
the full original spec and `dentalmine/README.md` for what's implemented.

## Status: Phase 1 complete (panoramic/OPG, end-to-end)

| | |
|---|---|
| ✅ Implemented | Full OPG pipeline: quality gate, Med-CLIP (MMKD→BiomedCLIP→Stub), router, preprocessor, panoramic head, **C3 per-tooth clustering**, **C4 ranked treatment prompts**, postprocessor, overlay renderer, CLI, 11 passing tests |
| ✅ Verified training data | `scripts/tufts_to_yolo.py` converts the Tufts dataset's real tooth boxes (Universal Numbering) to FDI-labeled YOLO format — coordinate transform verified both numerically and visually |
| 🚧 Not yet implemented | Bitewing/periapical/CBCT heads, MadClip C1/C2, nnU-Net 3D segmentation, DentVFM backbone, custom training phases, C4 Tier-2 VLM |

See:
- [`dentalmine/README.md`](dentalmine/README.md) — what's implemented, module-by-module
- [`A100_GUIDE.md`](A100_GUIDE.md) — environment setup, dataset downloads, running inference and training on a GPU server
- [`DATASETS.md`](DATASETS.md) — dataset inventory and organization
- [`PROMPT.md`](PROMPT.md) — original full project specification

## Quick start

```bash
cd dentalmine
pip install -e .
pip install open_clip_torch ultralytics pydicom   # real Med-CLIP + Stage-1 deps

# offline / deterministic (no model downloads, for quick sanity checks):
DENTALMIND_MEDCLIP=stub dentalmine infer \
  --input tests/fixtures/test_opg.jpg --modality opg --output ./results/ --device cpu

# real inference (BiomedCLIP auto-downloads ~430MB on first run):
dentalmine infer --input <your_opg.jpg> --modality opg --output ./results/
```

Outputs in `--output/`: `findings.json` (schema-validated), `original.png`,
`annotated.png` (color-coded overlay), `treatment_card.txt` (ranked per-tooth
treatment options + disclaimer), `summary.txt` (plain-language patient summary).

Run tests:

```bash
DENTALMIND_MEDCLIP=stub pytest dentalmine/tests/ -v
```

## Training (small datasets — no 10GB download required)

The Stage-1 FDI tooth detector trains on the Tufts Dental Database (~310MB,
real tooth bounding boxes) instead of the full 10GB DENTEX archive:

```bash
# download (see A100_GUIDE.md §2.3 for the Kaggle auth setup)
kaggle datasets download -d tommyngx/the-tufts-dental-database-2022 -p data/tufts_raw --unzip

# convert to YOLO format (FDI class names, verified transform)
cd dentalmine
python scripts/tufts_to_yolo.py \
  --bbox-json ../data/tufts_raw/Segmentation/teeth_bbox.json \
  --radiographs-dir ../data/tufts_raw/Radiographs \
  --out-dir ../data/tufts_yolo

# train (968 images, 32 FDI classes)
yolo detect train model=yolo11m.pt data=../data/tufts_yolo/data.yaml \
  epochs=150 imgsz=1024 batch=32 device=0
```

Full instructions (including the A100 server setup) are in `A100_GUIDE.md`.

## Repository layout

```
dentalmine/          # the installable package (cli.py, schema/, pipeline/, models/, tests/)
datasets/            # raw downloaded archives (gitignored)
data/                # organized/extracted + converted datasets (gitignored)
PROMPT.md            # original full project specification
DATASETS.md          # dataset inventory
A100_GUIDE.md         # GPU server setup, inference, training guide
```

## License / data

Datasets used (DENTEX, Tufts, Zenodo, Mendeley) each carry their own license —
see `DATASETS.md`. This repository contains code only; no dataset files are
committed (see `.gitignore`).
