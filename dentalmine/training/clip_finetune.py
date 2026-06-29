"""Fine-tune the shared Med-CLIP image tower on 2D dental data (prototype loss).

This adapts PROMPT.md's "shared backbone trained on all dental data" idea to what
is actually available right now: no free-text captions, just class labels (FDI
tooth numbers, cavity/healthy) from the converted 2D datasets, and no 3D data yet.

Design choice — prototype fine-tuning, not full bidirectional CLIP loss:
  With only ~34 distinct text prompts (32 FDI classes + cavity/healthy), a
  standard symmetric image<->text InfoNCE loss lets the text tower trivially
  memorize 34 fixed points (degenerate collapse) while teaching the image tower
  little. Instead: compute FROZEN text embeddings once per class (the "prototypes"),
  then fine-tune only the image tower with cross-entropy against those fixed
  prototypes — i.e. supervised contrastive fine-tuning using Med-CLIP's own
  zero-shot head as the classifier. This is the standard way to adapt a CLIP
  image encoder to a domain when you have labels but not captions.

Data sources (2D only — wire in 3D/CBCT here once available, see --data-root):
  tufts_yolo/      YOLO boxes -> tooth crops, label = FDI class (32 classes)
  caries_zenodo/   Pascal-VOC "cavity" boxes -> cavity crops (1 class) +
                    randomly sampled non-overlapping crops -> "healthy" class

Usage (run on the GPU server):
  python -m training.clip_finetune \
      --data-root ../data/2D \
      --output ../weights/medclip_dental_finetuned.pt \
      --epochs 10 --batch-size 64 --lr 1e-5
"""
from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.medclip.medclip_factory import MedCLIPFactory


@dataclass
class CropExample:
    image_path: Path
    bbox_xyxy: Tuple[float, float, float, float]  # pixel coords
    label: str
    split: str  # "train" | "val" — held out BY IMAGE, never by individual crop


def _load_tufts_yolo_examples(root: Path) -> Tuple[List[CropExample], List[str]]:
    """Respects Tufts' existing train/val image split (from tufts_to_yolo.py) —
    crops from a val-split image are never mixed into train, so the held-out
    set is genuinely unseen, not just a different crop of an image the model
    already saw."""
    data_yaml = root / "tufts_yolo" / "data.yaml"
    if not data_yaml.exists():
        return [], []
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = spec["names"]
    base = root / "tufts_yolo"
    examples = []
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                cls_idx, cx, cy, bw, bh = line.split()
                cls_idx = int(cls_idx)
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                examples.append(CropExample(
                    img_path, (x1, y1, x2, y2), f"fdi_{names[cls_idx]}", split,
                ))
    fdi_labels = [f"fdi_{n}" for n in names]
    return examples, fdi_labels


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_caries_zenodo_examples(
    root: Path, seed: int = 0, val_fraction: float = 0.1,
) -> List[CropExample]:
    """No pre-existing split here (unlike Tufts), so assign one deterministically
    PER IMAGE (not per crop) — every crop from a given x-ray goes to the same
    split, so cavity/healthy crops from the same image never leak across train/val."""
    base = root / "caries_zenodo" / "Dataset" / "x-ray"
    xml_dir, img_dir = base / "xmls", base / "images"
    if not xml_dir.exists():
        return []
    rng = random.Random(seed)
    xml_paths = sorted(xml_dir.glob("*.xml"))
    rng.shuffle(xml_paths)
    n_val = max(1, int(len(xml_paths) * val_fraction))
    val_stems = {p.stem for p in xml_paths[:n_val]}

    examples = []
    for xml_path in sorted(xml_dir.glob("*.xml")):
        img_path = img_dir / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            continue
        split = "val" if xml_path.stem in val_stems else "train"
        tree = ET.parse(xml_path)
        root_el = tree.getroot()
        size = root_el.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)
        boxes = []
        for obj in root_el.findall("object"):
            bb = obj.find("bndbox")
            box = (float(bb.find("xmin").text), float(bb.find("ymin").text),
                   float(bb.find("xmax").text), float(bb.find("ymax").text))
            boxes.append(box)
            examples.append(CropExample(img_path, box, "cavity", split))
        # one random non-overlapping crop per image as a "healthy" negative
        for _ in range(5):
            cw, ch = w * 0.15, h * 0.15
            x1 = rng.uniform(0, max(1.0, w - cw))
            y1 = rng.uniform(0, max(1.0, h - ch))
            cand = (x1, y1, x1 + cw, y1 + ch)
            if all(_iou(cand, b) < 0.05 for b in boxes):
                examples.append(CropExample(img_path, cand, "healthy", split))
                break
    return examples


class DentalCropDataset(Dataset):
    def __init__(self, examples: List[CropExample], label_to_idx: dict, preprocess):
        self.examples = examples
        self.label_to_idx = label_to_idx
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        with Image.open(ex.image_path) as im:
            crop = im.convert("RGB").crop(tuple(map(int, ex.bbox_xyxy)))
        tensor = self.preprocess(crop)
        return tensor, self.label_to_idx[ex.label]


_TEXT_TEMPLATES = {
    "cavity": "dental cavity lesion on a periapical X-ray",
    "healthy": "healthy tooth region on a dental X-ray",
}


def _label_to_prompt(label: str) -> str:
    if label.startswith("fdi_"):
        return f"tooth {label[4:]} on a dental panoramic radiograph"
    return _TEXT_TEMPLATES.get(label, label)


def build_dataset(data_root: Path, encoder):
    examples, _ = _load_tufts_yolo_examples(data_root)
    examples += _load_caries_zenodo_examples(data_root)
    if not examples:
        raise FileNotFoundError(
            f"No training examples found under {data_root}. Expected "
            f"{data_root}/tufts_yolo/ and/or {data_root}/caries_zenodo/."
        )
    labels = sorted({e.label for e in examples})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    train_examples = [e for e in examples if e.split == "train"]
    val_examples = [e for e in examples if e.split == "val"]
    train_ds = DentalCropDataset(train_examples, label_to_idx, encoder.preprocess)
    val_ds = DentalCropDataset(val_examples, label_to_idx, encoder.preprocess)
    return train_ds, val_ds, labels


def compute_prototypes(encoder, labels: List[str], device: str) -> torch.Tensor:
    prompts = [_label_to_prompt(l) for l in labels]
    with torch.no_grad():
        text_f = encoder.encode_text(prompts)  # [N, D], already L2-normalized
    return text_f.to(device)


def load_shared_encoder(args, device):
    """Load the shared Med-CLIP encoder (MMKD-CLIP -> BiomedCLIP) and verify it
    is trainable. Shared by the 2D fine-tune and the unified 2D+3D baseline."""
    from config_loader import load_config

    cfg = load_config(getattr(args, "config", None))
    print(f"Loading Med-CLIP encoder on {device}...")
    encoder = MedCLIPFactory.get(config=cfg, device=device, cache=False)
    if not hasattr(encoder, "model") or not hasattr(encoder, "tokenizer"):
        raise RuntimeError(
            f"Active encoder '{getattr(encoder, 'name', type(encoder))}' has no "
            f"trainable .model/.tokenizer (Stub encoder?). Set up MMKD-CLIP or "
            f"ensure BiomedCLIP/open_clip loaded correctly."
        )
    print(f"Active encoder: {encoder.name}")
    return encoder


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_shared_encoder(args, device)
    finetune_2d(encoder, args, device)


def finetune_2d(encoder, args, device):
    """Fine-tune the (shared) Med-CLIP image tower on 2D crops. Returns the
    trained encoder so a caller can reuse it for the 3D branch."""
    data_root = Path(args.data_root)
    train_ds, val_ds, labels = build_dataset(data_root, encoder)
    print(f"Dataset: {len(train_ds)} train crops, {len(val_ds)} held-out val crops "
          f"(split BY IMAGE, never by crop) across {len(labels)} classes: {labels}")
    if len(val_ds) == 0:
        print("WARNING: no held-out val crops — cannot check overfitting. "
              "Falling back to reporting train metrics only.", flush=True)

    prototypes = compute_prototypes(encoder, labels, device)  # frozen, [N, D]

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, drop_last=True)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        if len(val_ds) > 0 else None
    )

    model = encoder.model
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    # Freeze text tower explicitly — only the image tower is fine-tuned.
    text_module = getattr(model, "text", None) or getattr(model, "text_model", None)
    if text_module is not None:
        for p in text_module.parameters():
            p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0

    n_batches = len(loader)
    print(f"Starting training: {args.epochs} epochs x {n_batches} batches "
          f"(batch_size={args.batch_size}) on {device}", flush=True)

    import copy
    import time

    best_val_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        total_loss, n_correct, n_total = 0.0, 0, 0
        t_epoch = time.perf_counter()
        for batch_idx, (images, label_idx) in enumerate(loader):
            images = images.to(device)
            label_idx = label_idx.to(device)

            img_f = model.encode_image(images)
            img_f = F.normalize(img_f, dim=-1)
            logits = logit_scale * img_f @ prototypes.T

            loss = F.cross_entropy(logits, label_idx)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * images.size(0)
            n_correct += (logits.argmax(dim=-1) == label_idx).sum().item()
            n_total += images.size(0)

            if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.perf_counter() - t_epoch
                print(f"  epoch {epoch+1}/{args.epochs} batch {batch_idx+1}/{n_batches} "
                      f"loss={loss.item():.4f} running_acc={n_correct/n_total:.3f} "
                      f"elapsed={elapsed:.1f}s", flush=True)
            if getattr(args, "fast_dev_run", False):
                break

        train_acc = n_correct / n_total
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={total_loss/n_total:.4f}  "
              f"train_acc={train_acc:.3f}", flush=True)

        if val_loader is not None and not getattr(args, "fast_dev_run", False):
            val_loss, val_acc = evaluate(model, val_loader, prototypes, logit_scale, device)
            print(f"epoch {epoch+1}/{args.epochs}  HELD-OUT val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}  (gap vs train_acc: {train_acc - val_acc:+.3f})",
                  flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            model.train()

        if getattr(args, "fast_dev_run", False):
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save({"model_state_dict": final_state, "labels": labels,
                "encoder_name": encoder.name, "best_val_acc": best_val_acc}, out_path)
    if best_state is not None:
        print(f"Saved BEST checkpoint (val_acc={best_val_acc:.3f}) -> {out_path}")
    else:
        print(f"No val set was available — saved final-epoch weights -> {out_path}")
    return encoder


@torch.no_grad()
def evaluate(model, val_loader, prototypes, logit_scale, device) -> Tuple[float, float]:
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    for images, label_idx in val_loader:
        images = images.to(device)
        label_idx = label_idx.to(device)
        img_f = F.normalize(model.encode_image(images), dim=-1)
        logits = logit_scale * img_f @ prototypes.T
        loss = F.cross_entropy(logits, label_idx)
        total_loss += loss.item() * images.size(0)
        n_correct += (logits.argmax(dim=-1) == label_idx).sum().item()
        n_total += images.size(0)
    return total_loss / n_total, n_correct / n_total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/2D")
    ap.add_argument("--output", default="../weights/medclip_dental_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None, help="Path to config YAML (default: config/default.yaml)")
    train(ap.parse_args())
