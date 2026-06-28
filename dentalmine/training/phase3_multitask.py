"""Phase 3 — multitask fine-tuning of task heads on the frozen DentVFM backbone,
with a Med-CLIP alignment auxiliary loss (knowledge distillation into DentVFM).

Adapted to the data available now (2D crops): two heads share the frozen DentVFM
encoder — an FDI tooth classifier (Tufts crops) and a caries classifier
(caries_zenodo crops) — trained simultaneously. The auxiliary loss pulls a
projection of DentVFM features toward the (frozen) Med-CLIP image embedding of
the same crop, distilling Med-CLIP knowledge into the shared backbone
(PROMPT.md Phase-3 "alignment_loss_weight").

Held-out val split is by image (reuses the leak-free loaders from clip_finetune).
Runnable now:
  python -m training.phase3_multitask --data-root ../data/2D --fast-dev-run \
      --model vit_base_patch16_224 --img-size 224
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.backbone.dentvfm import DentVFMEncoder
from models.medclip.medclip_factory import MedCLIPFactory
from training.clip_finetune import (
    _load_caries_zenodo_examples, _load_tufts_yolo_examples,
)
from training.trainer_base import resolve_device, save_checkpoint


def _task_of(label: str) -> int:
    return 0 if label.startswith("fdi_") else 1   # 0=tooth, 1=caries


class MultiTaskCropDataset(Dataset):
    def __init__(self, examples, fdi_to_idx, caries_to_idx, img_size, clip_preprocess):
        self.examples = examples
        self.fdi_to_idx = fdi_to_idx
        self.caries_to_idx = caries_to_idx
        self.dentvfm_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        with Image.open(ex.image_path) as im:
            crop = im.convert("RGB").crop(tuple(map(int, ex.bbox_xyxy)))
        task = _task_of(ex.label)
        label = self.fdi_to_idx[ex.label] if task == 0 else self.caries_to_idx[ex.label]
        return self.dentvfm_tf(crop), self.clip_preprocess(crop), task, label


def _build_examples(data_root: Path):
    examples, _ = _load_tufts_yolo_examples(data_root)
    examples += _load_caries_zenodo_examples(data_root)
    if not examples:
        raise FileNotFoundError(f"No 2D crops found under {data_root}.")
    fdi = sorted({e.label for e in examples if e.label.startswith("fdi_")})
    caries = sorted({e.label for e in examples if not e.label.startswith("fdi_")})
    return examples, {l: i for i, l in enumerate(fdi)}, {l: i for i, l in enumerate(caries)}


@torch.no_grad()
def _evaluate(backbone, heads, loader, device):
    backbone_eval_correct = {0: 0, 1: 0}
    totals = {0: 0, 1: 0}
    for dv, _clip, task, label in loader:
        dv, task, label = dv.to(device), task.to(device), label.to(device)
        feats = backbone.encode_cls(dv)
        for t in (0, 1):
            m = task == t
            if m.any():
                pred = heads[t](feats[m]).argmax(dim=-1)
                backbone_eval_correct[t] += (pred == label[m]).sum().item()
                totals[t] += int(m.sum())
    return {
        "tooth_acc": backbone_eval_correct[0] / max(1, totals[0]),
        "caries_acc": backbone_eval_correct[1] / max(1, totals[1]),
    }


def train(args):
    device = resolve_device(args.device)
    data_root = Path(args.data_root)
    examples, fdi_to_idx, caries_to_idx = _build_examples(data_root)

    medclip = MedCLIPFactory.get(config=None, device=device, cache=False)
    if not hasattr(medclip, "preprocess"):
        raise RuntimeError(f"Med-CLIP encoder '{getattr(medclip,'name','?')}' has no "
                           f".preprocess (Stub?). Need real BiomedCLIP/MMKD for alignment.")

    train_ex = [e for e in examples if e.split == "train"]
    val_ex = [e for e in examples if e.split == "val"]
    train_ds = MultiTaskCropDataset(train_ex, fdi_to_idx, caries_to_idx, args.img_size, medclip.preprocess)
    val_ds = MultiTaskCropDataset(val_ex, fdi_to_idx, caries_to_idx, args.img_size, medclip.preprocess)
    print(f"Phase 3: {len(train_ds)} train / {len(val_ds)} val crops; "
          f"{len(fdi_to_idx)} FDI + {len(caries_to_idx)} caries classes; device={device}", flush=True)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers) if len(val_ds) else None

    backbone = DentVFMEncoder(model_name=args.model, weights_path=args.backbone_ckpt,
                              pretrained=True, freeze=True, device=device, img_size=args.img_size)
    d = backbone.embed_dim
    tooth_head = nn.Linear(d, len(fdi_to_idx)).to(device)
    caries_head = nn.Linear(d, len(caries_to_idx)).to(device)
    heads = {0: tooth_head, 1: caries_head}
    align_proj = nn.Linear(d, medclip.image_embed_dim).to(device)

    params = list(tooth_head.parameters()) + list(caries_head.parameters()) + list(align_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    w_align = args.alignment_loss_weight

    n_batches = len(loader)
    print(f"Starting: {args.epochs} epochs x {n_batches} batches", flush=True)
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        for bi, (dv, clip_t, task, label) in enumerate(loader):
            dv, clip_t = dv.to(device), clip_t.to(device)
            task, label = task.to(device), label.to(device)

            with torch.no_grad():
                feats = backbone.encode_cls(dv)                    # frozen backbone
                clip_feats = medclip.encode_image(clip_t)          # frozen Med-CLIP

            loss = torch.zeros((), device=device)
            for t in (0, 1):
                m = task == t
                if m.any():
                    loss = loss + F.cross_entropy(heads[t](feats[m]), label[m])
            align = F.mse_loss(F.normalize(align_proj(feats), dim=-1), clip_feats)
            loss = loss + w_align * align

            opt.zero_grad(); loss.backward(); opt.step()
            if bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
                print(f"  epoch {epoch+1}/{args.epochs} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f} align={align.item():.4f} "
                      f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
            if args.fast_dev_run:
                break

        if val_loader is not None and not args.fast_dev_run:
            metrics = _evaluate(backbone, heads, val_loader, device)
            print(f"epoch {epoch+1}/{args.epochs} HELD-OUT "
                  f"tooth_acc={metrics['tooth_acc']:.3f} caries_acc={metrics['caries_acc']:.3f}",
                  flush=True)
        if args.fast_dev_run:
            break

    save_checkpoint(args.output,
                    tooth_head=tooth_head.state_dict(), caries_head=caries_head.state_dict(),
                    align_proj=align_proj.state_dict(),
                    fdi_labels=list(fdi_to_idx), caries_labels=list(caries_to_idx),
                    phase="phase3_multitask")
    print(f"Saved multitask heads -> {args.output}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/2D")
    ap.add_argument("--output", default="../weights/phase3_heads.pt")
    ap.add_argument("--backbone-ckpt", default=None, help="DentVFM weights from Phase 1.")
    ap.add_argument("--model", default="vit_large_patch14_dinov2")
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alignment-loss-weight", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
